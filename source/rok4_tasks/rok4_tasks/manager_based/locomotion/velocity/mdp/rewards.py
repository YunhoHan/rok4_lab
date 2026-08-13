"""Reward terms for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from rok4_tasks.assets.robots.rok4 import ROK4_JOINT_ORDER

from .observations import _adapt_actuator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROK4_RELAXED_ACTION_IDS = (2, 3, 8, 9)
"""Actuator indices with reduced physical effort and state penalties."""


def _log_episode_event_average(
    env: ManagerBasedRLEnv,
    key: str,
    value_sum: torch.Tensor,
    event_count: torch.Tensor,
    env_ids: Sequence[int] | slice,
) -> None:
    """Log an event-weighted episode average without synchronizing to the CPU."""
    if not hasattr(env, "extras"):
        return

    selected_count = torch.sum(event_count[env_ids])
    selected_sum = torch.sum(value_sum[env_ids])
    average = torch.where(
        selected_count > 0,
        selected_sum / selected_count.clamp_min(1.0),
        torch.zeros_like(selected_sum),
    )
    env.extras.setdefault("log", {})[key] = average


def feet_air_time_touchdown_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_air_time: float,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Compute the stateless RoK4-local touchdown air-time reward."""
    if command_threshold < 0.0:
        raise ValueError(f"command_threshold must be non-negative, received {command_threshold}.")
    if target_air_time < 0.0:
        raise ValueError(f"target_air_time must be non-negative, received {target_air_time}.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    single_touchdown = torch.sum(first_contact.int(), dim=1) == 1
    moving_command = (
        torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
        > command_threshold
    )
    valid_touchdown = first_contact & single_touchdown.unsqueeze(-1) & moving_command.unsqueeze(-1)
    return torch.sum((last_air_time - target_air_time) * valid_touchdown, dim=1)


class FeetAirTimeTouchdownBiped(ManagerTermBase):
    """Shape completed swing time and report mean rewarded touchdown air time [s]."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-environment touchdown air-time statistics."""
        super().__init__(cfg, env)
        target_air_time = cfg.params["target_air_time"]
        command_threshold = cfg.params.get("command_threshold", 0.1)
        self._validate_parameters(target_air_time, command_threshold)

        self._air_time_sum = torch.zeros(env.num_envs, device=env.device)
        self._touchdown_count = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        target_air_time: float,
        sensor_cfg: SceneEntityCfg,
        command_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Compute touchdown air-time shaping and accumulate valid event times."""
        self._validate_parameters(target_air_time, command_threshold)

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
        single_touchdown = torch.sum(first_contact.int(), dim=1) == 1
        moving_command = (
            torch.linalg.vector_norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
            > command_threshold
        )
        valid_touchdown = first_contact & single_touchdown.unsqueeze(-1) & moving_command.unsqueeze(-1)
        reward = torch.sum((last_air_time - target_air_time) * valid_touchdown, dim=1)

        self._air_time_sum += torch.sum(last_air_time * valid_touchdown, dim=1)
        self._touchdown_count += torch.sum(valid_touchdown, dim=1)
        return reward

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log and clear valid touchdown air-time statistics."""
        if env_ids is None:
            env_ids = slice(None)

        _log_episode_event_average(
            self._env,
            "Metrics/feet_touchdown/mean_air_time",
            self._air_time_sum,
            self._touchdown_count,
            env_ids,
        )
        self._air_time_sum[env_ids] = 0.0
        self._touchdown_count[env_ids] = 0.0

    @staticmethod
    def _validate_parameters(
        target_air_time: float,
        command_threshold: float,
    ) -> None:
        """Validate touchdown timing and command-gating parameters."""
        if command_threshold < 0.0:
            raise ValueError(f"command_threshold must be non-negative, received {command_threshold}.")
        if target_air_time < 0.0:
            raise ValueError(f"target_air_time must be non-negative, received {target_air_time}.")


def feet_touchdown_acc(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 50.0,
) -> torch.Tensor:
    """Penalize excessive foot linear acceleration at first contact [m/s^2]."""
    if threshold < 0.0:
        raise ValueError(f"threshold must be non-negative, received {threshold}.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    foot_acc = torch.linalg.vector_norm(
        asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :],
        dim=-1,
    )
    if foot_acc.shape != first_contact.shape:
        raise ValueError(
            f"Selected foot acceleration shape {foot_acc.shape} does not match "
            f"first-contact shape {first_contact.shape}."
        )

    impact_acc = torch.clamp(foot_acc - threshold, min=0.0)
    return torch.sum(impact_acc * first_contact, dim=1)


class FeetTouchdownVelocityL2(ManagerTermBase):
    """Penalize excessive pre-touchdown downward foot speed [(m/s)^2].

    The term stores each selected foot's world-Z velocity from the previous
    policy step. When first contact is reported, it penalizes only the squared
    amount by which the preceding downward speed exceeded
    :paramref:`safe_landing_velocity`. This event-only formulation does not
    shape the airborne approach or established stance.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-foot velocity history."""
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        safe_landing_velocity = cfg.params.get("safe_landing_velocity", 0.0)
        self._validate_safe_landing_velocity(safe_landing_velocity)

        num_sensor_bodies = len(sensor_cfg.body_ids)
        num_asset_bodies = len(asset_cfg.body_ids)
        if num_sensor_bodies != num_asset_bodies:
            raise ValueError(
                f"Selected {num_asset_bodies} foot bodies but received "
                f"{num_sensor_bodies} contact bodies."
            )

        shape = (env.num_envs, num_asset_bodies)
        self._previous_foot_vel_z = torch.zeros(shape, device=env.device)
        self._previous_in_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self._has_previous_sample = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self._touchdown_speed_sum = torch.zeros(env.num_envs, device=env.device)
        self._touchdown_count = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        safe_landing_velocity: float = 0.0,
    ) -> torch.Tensor:
        """Compute one touchdown penalty and update foot velocity history."""
        self._validate_safe_landing_velocity(safe_landing_velocity)

        asset = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
        in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

        if (
            foot_vel_z.shape != self._previous_foot_vel_z.shape
            or foot_vel_z.shape != in_contact.shape
            or foot_vel_z.shape != first_contact.shape
        ):
            raise ValueError(
                f"Selected foot velocity shape {foot_vel_z.shape} does not match stored history "
                f"{self._previous_foot_vel_z.shape} and contact shapes "
                f"{in_contact.shape} and {first_contact.shape}."
            )

        valid_touchdown = first_contact & self._has_previous_sample & (~self._previous_in_contact)
        pre_touchdown_speed = torch.relu(-self._previous_foot_vel_z)
        speed_excess = torch.relu(pre_touchdown_speed - safe_landing_velocity)
        penalty = torch.sum(torch.square(speed_excess) * valid_touchdown, dim=1)

        touchdown_count = torch.sum(valid_touchdown, dim=1)
        self._touchdown_speed_sum += torch.sum(pre_touchdown_speed * valid_touchdown, dim=1)
        self._touchdown_count += touchdown_count

        self._previous_foot_vel_z.copy_(foot_vel_z)
        self._previous_in_contact.copy_(in_contact)
        self._has_previous_sample.fill_(True)
        return penalty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear stored velocity history for completed environments."""
        if env_ids is None:
            env_ids = slice(None)

        _log_episode_event_average(
            self._env,
            "Metrics/feet_touchdown/mean_pre_touchdown_vertical_speed",
            self._touchdown_speed_sum,
            self._touchdown_count,
            env_ids,
        )
        self._previous_foot_vel_z[env_ids] = 0.0
        self._previous_in_contact[env_ids] = False
        self._has_previous_sample[env_ids] = False
        self._touchdown_speed_sum[env_ids] = 0.0
        self._touchdown_count[env_ids] = 0.0

    @staticmethod
    def _validate_safe_landing_velocity(safe_landing_velocity: float) -> None:
        """Validate the permitted pre-touchdown downward speed [m/s]."""
        if safe_landing_velocity < 0.0:
            raise ValueError(
                "safe_landing_velocity must be non-negative, received "
                f"{safe_landing_velocity}."
            )


def feet_contact_velocity_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    landing_height: float = 0.03,
    approach_velocity_threshold: float = -0.6,
    impact_velocity_threshold: float = 0.0,
) -> torch.Tensor:
    """Penalize excessive downward foot speed near and at touchdown [(m/s)^2].

    The proactive component acts before contact when a foot is below
    :paramref:`landing_height` and descending faster than
    :paramref:`approach_velocity_threshold`. The reactive component acts when
    the contact sensor reports first contact during the policy step.
    """
    if landing_height < 0.0:
        raise ValueError(f"landing_height must be non-negative, received {landing_height}.")
    if approach_velocity_threshold > 0.0:
        raise ValueError(
            "approach_velocity_threshold must be non-positive, received "
            f"{approach_velocity_threshold}."
        )
    if impact_velocity_threshold > 0.0:
        raise ValueError(
            "impact_velocity_threshold must be non-positive, received "
            f"{impact_velocity_threshold}."
        )

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    foot_height = foot_pos_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    if foot_vel_z.shape != in_contact.shape or foot_vel_z.shape != first_contact.shape:
        raise ValueError(
            f"Selected foot velocity shape {foot_vel_z.shape} does not match contact shapes "
            f"{in_contact.shape} and {first_contact.shape}."
        )

    approaching_too_fast = (
        (~in_contact)
        & (foot_height < landing_height)
        & (foot_vel_z < approach_velocity_threshold)
    )
    hard_landing = first_contact & (foot_vel_z < impact_velocity_threshold)
    penalty_mask = approaching_too_fast | hard_landing
    return torch.sum(torch.square(foot_vel_z) * penalty_mask, dim=1)


def feet_contact_force_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 1000.0,
    stance_weight: float = 0.15,
    landing_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize excessive foot normal force using a hybrid stance/landing mask.

    The maximum normal-force magnitude across the contact sensor history is
    used so a physics-substep impact peak is retained until reward evaluation.
    """
    if max_contact_force <= 0.0:
        raise ValueError(f"max_contact_force must be positive, received {max_contact_force}.")
    if stance_weight < 0.0 or landing_weight < 0.0:
        raise ValueError(
            "stance_weight and landing_weight must be non-negative, received "
            f"{stance_weight} and {landing_weight}."
        )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
    ground_force_history = torch.nan_to_num(force_history).sum(dim=3)
    peak_force = torch.linalg.vector_norm(ground_force_history, dim=-1).max(dim=1).values
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    if peak_force.shape != in_contact.shape or peak_force.shape != first_contact.shape:
        raise ValueError(
            f"Selected peak-force shape {peak_force.shape} does not match contact shapes "
            f"{in_contact.shape} and {first_contact.shape}."
        )

    contact_weight = stance_weight * in_contact * (~first_contact)
    contact_weight += landing_weight * first_contact
    force_excess_ratio = torch.relu(peak_force - max_contact_force) / max_contact_force
    return torch.sum(torch.square(force_excess_ratio) * contact_weight, dim=1)


class FeetContactForceL2(ManagerTermBase):
    """Penalize the peak force in an initial landing window and report its event average [N]."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize landing-window state and touchdown-force statistics."""
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        force_limit_multiplier = cfg.params.get("force_limit_multiplier", 1.5)
        landing_window_s = cfg.params.get("landing_window_s", 0.05)
        self._validate_parameters(force_limit_multiplier, landing_window_s)

        self._touchdown_force_sum = torch.zeros(env.num_envs, device=env.device)
        self._touchdown_count = torch.zeros(env.num_envs, device=env.device)
        self._num_feet = len(sensor_cfg.body_ids)
        shape = (env.num_envs, self._num_feet)
        self._landing_active = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self._landing_peak_force = torch.zeros(shape, device=env.device)
        # Startup mass DR runs after Reward Manager construction, so read randomized masses on first use.
        self._robot_weight: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        force_limit_multiplier: float = 1.5,
        landing_window_s: float = 0.05,
    ) -> torch.Tensor:
        """Complete each landing window and penalize its maximum force once."""
        self._validate_parameters(force_limit_multiplier, landing_window_s)

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        force_history = contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
        ground_force_history = torch.nan_to_num(force_history).sum(dim=3)
        peak_force = torch.linalg.vector_norm(ground_force_history, dim=-1).max(dim=1).values
        contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
        in_contact = contact_time > 0.0
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

        expected_shape = (env.num_envs, self._num_feet)
        if peak_force.shape != expected_shape or contact_time.shape != expected_shape:
            raise ValueError(
                f"Selected peak-force/contact-time shapes {peak_force.shape}/{contact_time.shape} "
                f"do not match expected shape {expected_shape}."
            )

        self._landing_active |= first_contact
        self._landing_peak_force = torch.where(
            self._landing_active,
            torch.maximum(self._landing_peak_force, peak_force),
            self._landing_peak_force,
        )
        landing_complete = self._landing_active & ((contact_time >= landing_window_s) | (~in_contact))

        if self._robot_weight is None:
            asset = env.scene[asset_cfg.name]
            masses = asset.root_physx_view.get_masses().to(
                device=self._landing_peak_force.device,
                dtype=self._landing_peak_force.dtype,
            )
            gravity = torch.as_tensor(env.sim.cfg.gravity, device=masses.device, dtype=masses.dtype)
            gravity_magnitude = torch.linalg.vector_norm(gravity)
            if gravity_magnitude <= 0.0:
                raise ValueError("FeetContactForceL2 requires non-zero gravity.")
            self._robot_weight = torch.sum(masses, dim=1) * gravity_magnitude

        force_limit = force_limit_multiplier * self._robot_weight.unsqueeze(-1)
        force_excess_ratio = torch.relu(self._landing_peak_force - force_limit) / force_limit
        penalty = torch.sum(torch.square(force_excess_ratio) * landing_complete, dim=1)

        self._touchdown_force_sum += torch.sum(self._landing_peak_force * landing_complete, dim=1)
        self._touchdown_count += torch.sum(landing_complete, dim=1)
        self._landing_active &= ~landing_complete
        self._landing_peak_force.masked_fill_(landing_complete, 0.0)
        return penalty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log and clear touchdown-force statistics for completed environments."""
        if env_ids is None:
            env_ids = slice(None)

        _log_episode_event_average(
            self._env,
            "Metrics/feet_touchdown/mean_peak_normal_force",
            self._touchdown_force_sum,
            self._touchdown_count,
            env_ids,
        )
        self._touchdown_force_sum[env_ids] = 0.0
        self._touchdown_count[env_ids] = 0.0
        self._landing_active[env_ids] = False
        self._landing_peak_force[env_ids] = 0.0

    @staticmethod
    def _validate_parameters(force_limit_multiplier: float, landing_window_s: float) -> None:
        """Validate landing-window and body-weight scaling parameters."""
        if force_limit_multiplier <= 0.0:
            raise ValueError(
                f"force_limit_multiplier must be positive, received {force_limit_multiplier}."
            )
        if landing_window_s <= 0.0:
            raise ValueError(f"landing_window_s must be positive, received {landing_window_s}.")


def base_height_relative_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height error above the flat environment origin [m^2]."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.square(height - target_height)


def feet_swing_clearance_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    std: float,
    velocity_scale: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward command-directed swing feet near a target flat-ground clearance.

    The height is the selected foot-body origin's world-Z coordinate relative
    to the environment origin. For linear commands, the velocity gate uses only
    foot progress along the commanded planar direction. Pure-yaw commands retain
    the speed-magnitude gate because their two feet move in opposite directions.
    The term is disabled in standing environments.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, received {std}.")
    if velocity_scale <= 0.0:
        raise ValueError(f"velocity_scale must be positive, received {velocity_scale}.")

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    if foot_pos_w.shape[1] != current_air_time.shape[1]:
        raise ValueError(
            f"Selected {foot_pos_w.shape[1]} foot bodies but received "
            f"{current_air_time.shape[1]} contact bodies."
        )

    foot_height = foot_pos_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    root_yaw_w = yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, foot_vel_w.shape[1], -1)
    foot_vel_yaw = quat_apply_inverse(root_yaw_w, foot_vel_w)
    foot_xy_speed = torch.linalg.vector_norm(foot_vel_yaw[..., :2], dim=-1)

    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_xy_norm = torch.linalg.vector_norm(command_xy, dim=-1)
    command_xy_dir = command_xy / command_xy_norm.clamp_min(torch.finfo(command.dtype).eps).unsqueeze(-1)
    foot_progress_speed = torch.relu(
        torch.sum(foot_vel_yaw[..., :2] * command_xy_dir.unsqueeze(1), dim=-1)
    )
    swing_speed = torch.where(command_xy_norm.unsqueeze(-1) > 0.0, foot_progress_speed, foot_xy_speed)
    velocity_gate = torch.tanh(swing_speed / velocity_scale)
    height_reward = torch.exp(-torch.square(foot_height - target_height) / std**2)

    command_term = env.command_manager.get_term(command_name)
    valid_swing = current_air_time > 0.0
    valid_swing &= ~command_term.is_standing_env.unsqueeze(-1)
    valid_count = torch.sum(valid_swing, dim=1)
    reward = torch.sum(velocity_gate * height_reward * valid_swing, dim=1)
    return torch.where(valid_count > 0, reward / valid_count.clamp(min=1), torch.zeros_like(reward))


def feet_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the tilt of feet that are in contact with the ground."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    foot_up_local = torch.zeros_like(foot_quat_w[..., :3])
    foot_up_local[..., 2] = 1.0
    foot_up_w = quat_apply(foot_quat_w, foot_up_local)
    tilt_error = torch.sum(torch.square(foot_up_w[..., :2]), dim=-1)

    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    contact_count = torch.sum(in_contact, dim=1).clamp(min=1)
    return torch.sum(tilt_error * in_contact, dim=1) / contact_count


def _feet_swing_tilt_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    axis: int,
) -> torch.Tensor:
    """Penalize one yaw-removed sole-normal component while each foot is in swing."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    foot_up_local = torch.zeros_like(foot_quat_w[..., :3])
    foot_up_local[..., 2] = 1.0
    foot_up_w = quat_apply(foot_quat_w, foot_up_local)
    foot_up_yaw = quat_apply_inverse(
        yaw_quat(foot_quat_w),
        foot_up_w,
    )
    tilt_error = torch.square(foot_up_yaw[..., axis])

    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if in_contact.shape[1] != tilt_error.shape[1]:
        raise ValueError(
            f"Selected {tilt_error.shape[1]} foot bodies but received "
            f"{in_contact.shape[1]} contact bodies."
        )

    in_swing = ~in_contact
    swing_count = torch.sum(in_swing, dim=1).clamp(min=1)
    return torch.sum(tilt_error * in_swing, dim=1) / swing_count


def feet_swing_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize lateral sole tilt only while each foot is in swing."""
    return _feet_swing_tilt_l2(env, asset_cfg, sensor_cfg, axis=1)


def feet_swing_pitch_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize sagittal sole tilt only while each foot is in swing."""
    return _feet_swing_tilt_l2(env, asset_cfg, sensor_cfg, axis=0)


def feet_stance_width_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_width: float,
    hard_min_width: float,
    wide_coeff: float,
    hard_narrow_coeff: float,
    asset_cfg: SceneEntityCfg,
    lateral_command_threshold: float = 0.05,
    yaw_command_threshold: float = 0.05,
    moving_command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize an excessively wide or crossed stance during straight walking."""
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_pos_w.shape[1] != 2:
        raise ValueError(f"Expected exactly two feet, received {foot_pos_w.shape[1]} bodies.")

    left_to_right_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
    left_to_right_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), left_to_right_w)
    stance_width = left_to_right_yaw[:, 1]

    width_too_wide = torch.relu(stance_width - target_width)
    width_too_narrow = torch.relu(hard_min_width - stance_width)
    penalty = wide_coeff * torch.square(width_too_wide)
    penalty += hard_narrow_coeff * torch.square(width_too_narrow)

    command = env.command_manager.get_command(command_name)
    straight_command = (torch.abs(command[:, 1]) <= lateral_command_threshold) & (
        torch.abs(command[:, 2]) <= yaw_command_threshold
    )
    moving_command = torch.linalg.vector_norm(command, dim=1) > moving_command_threshold
    return penalty * straight_command * moving_command


def feet_lateral_separation_l2(
    env: ManagerBasedRLEnv,
    minimum_width: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize insufficient signed lateral foot separation [m^2]."""
    asset = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if foot_pos_w.shape[1] != 2:
        raise ValueError(f"Expected exactly two feet, received {foot_pos_w.shape[1]} bodies.")

    left_to_right_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
    left_to_right_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), left_to_right_w)
    signed_width = left_to_right_yaw[:, 1]
    width_deficit = torch.relu(minimum_width - signed_width)
    return torch.square(width_deficit)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize absolute joint-position deviations in designated standing environments [rad]."""
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    joint_pos_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(joint_pos_error), dim=1) * command_term.is_standing_env


def _weighted_l2(values: torch.Tensor) -> torch.Tensor:
    """Compute weighted squared sum."""
    weights = torch.ones_like(values)
    weights[:, list(ROK4_RELAXED_ACTION_IDS)] = 0.5
    return torch.sum(torch.square(values) * weights, dim=1)


def actuator_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Penalize applied actuator torques with the RoK4 actuator weights."""
    asset = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    return _weighted_l2(actuator.applied_actuator_effort)


def actuator_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Penalize actuator velocities with the RoK4 actuator weights."""
    asset = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    actuator_vel = actuator.transmission.joint_to_actuator_velocity(joint_vel)
    return _weighted_l2(actuator_vel)


def _raw_action_l2(action_delta: torch.Tensor) -> torch.Tensor:
    """Compute the weighted squared difference of raw policy actions."""
    return _weighted_l2(action_delta)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order raw-action changes."""
    action_rate = env.action_manager.action - env.action_manager.prev_action
    penalty = _raw_action_l2(action_rate)
    has_history = torch.any(env.action_manager.prev_action != 0.0, dim=1)
    return torch.where(has_history, penalty, torch.zeros_like(penalty))


def actuator_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Penalize physical actuator accelerations using RoK4 actuator weights."""
    asset = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    actuator_acc = actuator.transmission.joint_to_actuator_acceleration(joint_acc)
    return _weighted_l2(actuator_acc)


def actuator_velocity_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Penalize actuator velocity beyond the configured limits."""
    asset = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    actuator_vel = actuator.transmission.joint_to_actuator_velocity(joint_vel)
    limit_excess = torch.relu(torch.abs(actuator_vel) - actuator.actuator_velocity_limit)
    return torch.sum(limit_excess, dim=1)


def actuator_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Penalize requested actuator torque beyond the configured limits."""
    asset = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    limit_excess = torch.relu(
        torch.abs(actuator.computed_actuator_effort) - actuator.actuator_torque_limit
    )
    return torch.sum(limit_excess, dim=1)


def joint_action_target_pos_limits(
    env: ManagerBasedRLEnv,
    action_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint-position action targets that cross the soft position limits."""
    asset = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    target_pos = action_term.processed_actions
    soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]

    if target_pos.shape[1] != soft_limits.shape[1]:
        raise ValueError(
            f"Action term '{action_name}' has {target_pos.shape[1]} targets, but the reward selected "
            f"{soft_limits.shape[1]} joints."
        )

    below_lower = (soft_limits[..., 0] - target_pos).clip(min=0.0)
    above_upper = (target_pos - soft_limits[..., 1]).clip(min=0.0)
    return torch.sum(below_lower + above_upper, dim=1)


class second_action_rate_l2(ManagerTermBase):
    """Penalize second-order raw-action changes."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the stateful reward term.

        Args:
            cfg: Reward term configuration.
            env: Manager-based RL environment.
        """
        super().__init__(cfg, env)
        self._prev_prev_action = torch.zeros_like(env.action_manager.action)
        self._action_history_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset stored action history for selected environments.

        Args:
            env_ids: Environment ids to reset. Defaults to all environments.
        """
        if env_ids is None:
            env_ids = slice(None)

        self._prev_prev_action[env_ids] = 0.0
        self._action_history_count[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the squared second-order action difference."""
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        action_2nd_rate = action - 2.0 * prev_action + self._prev_prev_action
        penalty = _raw_action_l2(action_2nd_rate)

        penalty = torch.where(self._action_history_count >= 2, penalty, torch.zeros_like(penalty))
        self._prev_prev_action[:] = prev_action
        self._action_history_count[:] = torch.clamp(self._action_history_count + 1, max=2)
        return penalty
