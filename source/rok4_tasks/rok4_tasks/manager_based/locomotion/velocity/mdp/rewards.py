"""Reward terms for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from rok4_tasks.assets.robots.rok4 import ROK4_JOINT_ORDER

from .observations import _adapt_actuator

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
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


def feet_standing_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count missing foot contacts at exactly zero velocity command.

    Apply a negative reward weight to penalize single support and flight during
    standing. No default joint pose, foot placement, load balance, or recovery
    gate is imposed. Recovery steps at zero command also incur this cost.

    Args:
        env: Environment providing velocity commands and contact sensor data.
        command_name: Three-component command: XY velocity [m/s] and yaw rate [rad/s].
        sensor_cfg: Sensor selection containing exactly two feet, with air-time tracking enabled.

    Returns:
        Missing-contact count (0, 1, or 2) per environment, or zero for any
        nonzero command, including pure yaw. Shape is (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if not contact_sensor.cfg.track_air_time:
        raise ValueError("feet_standing_contact requires track_air_time=True on the contact sensor.")
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    if contact_time.shape[1] != 2:
        raise ValueError(f"Expected exactly two feet, received {contact_time.shape[1]} bodies.")

    in_contact = contact_time > 0.0
    command = env.command_manager.get_command(command_name)
    standing = torch.all(command[:, :3] == 0.0, dim=1)
    missing_contacts = torch.sum((~in_contact).to(contact_time.dtype), dim=1)
    return missing_contacts * standing


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
    """Penalize pre-touchdown planar and downward foot COM velocity [(m/s)^2].

    The term stores each selected foot's COM world-frame velocity
    from the previous policy step, not link-origin or sole-point velocity.
    When first contact is reported, it penalizes the preceding
    planar speed and the squared amount by which the preceding downward speed
    exceeded :paramref:`safe_landing_velocity`. This event-only formulation
    does not shape the airborne approach or established stance.
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

        event_shape = (env.num_envs, num_asset_bodies)
        self._previous_foot_vel_w = torch.zeros((*event_shape, 3), device=env.device)
        self._previous_in_contact = torch.zeros(event_shape, dtype=torch.bool, device=env.device)
        self._has_previous_sample = torch.zeros(event_shape, dtype=torch.bool, device=env.device)
        self._touchdown_planar_speed_sum = torch.zeros(env.num_envs, device=env.device)
        self._touchdown_vertical_speed_sum = torch.zeros(env.num_envs, device=env.device)
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
        foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :3]
        in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

        if (
            foot_vel_w.shape != self._previous_foot_vel_w.shape
            or foot_vel_w.shape[:2] != in_contact.shape
            or foot_vel_w.shape[:2] != first_contact.shape
        ):
            raise ValueError(
                f"Selected foot velocity shape {foot_vel_w.shape} does not match stored history "
                f"{self._previous_foot_vel_w.shape} and contact shapes "
                f"{in_contact.shape} and {first_contact.shape}."
            )

        valid_touchdown = first_contact & self._has_previous_sample & (~self._previous_in_contact)
        pre_touchdown_planar_speed = torch.linalg.vector_norm(
            self._previous_foot_vel_w[..., :2],
            dim=-1,
        )
        pre_touchdown_vertical_speed = torch.relu(-self._previous_foot_vel_w[..., 2])
        vertical_speed_excess = torch.relu(
            pre_touchdown_vertical_speed - safe_landing_velocity
        )
        penalty_per_foot = torch.square(pre_touchdown_planar_speed)
        penalty_per_foot += torch.square(vertical_speed_excess)
        penalty = torch.sum(penalty_per_foot * valid_touchdown, dim=1)

        touchdown_count = torch.sum(valid_touchdown, dim=1)
        self._touchdown_planar_speed_sum += torch.sum(
            pre_touchdown_planar_speed * valid_touchdown,
            dim=1,
        )
        self._touchdown_vertical_speed_sum += torch.sum(
            pre_touchdown_vertical_speed * valid_touchdown,
            dim=1,
        )
        self._touchdown_count += touchdown_count

        self._previous_foot_vel_w.copy_(foot_vel_w)
        self._previous_in_contact.copy_(in_contact)
        self._has_previous_sample.fill_(True)
        return penalty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear stored velocity history for completed environments."""
        if env_ids is None:
            env_ids = slice(None)

        _log_episode_event_average(
            self._env,
            "Metrics/feet_touchdown/mean_pre_touchdown_planar_speed",
            self._touchdown_planar_speed_sum,
            self._touchdown_count,
            env_ids,
        )
        _log_episode_event_average(
            self._env,
            "Metrics/feet_touchdown/mean_pre_touchdown_vertical_speed",
            self._touchdown_vertical_speed_sum,
            self._touchdown_count,
            env_ids,
        )
        self._previous_foot_vel_w[env_ids] = 0.0
        self._previous_in_contact[env_ids] = False
        self._has_previous_sample[env_ids] = False
        self._touchdown_planar_speed_sum[env_ids] = 0.0
        self._touchdown_vertical_speed_sum[env_ids] = 0.0
        self._touchdown_count[env_ids] = 0.0

    @staticmethod
    def _validate_safe_landing_velocity(safe_landing_velocity: float) -> None:
        """Validate the permitted pre-touchdown downward speed [m/s]."""
        if safe_landing_velocity < 0.0:
            raise ValueError(
                "safe_landing_velocity must be non-negative, received "
                f"{safe_landing_velocity}."
            )


def _sole_point_kinematics(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    sole_points_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct sole-point positions and velocities from the same link reference."""
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    body_link_lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids]
    point_shape = (*body_pos_w.shape[:2], sole_points_b.shape[0], 3)
    point_offsets_b = sole_points_b.view(1, 1, -1, 3).expand(point_shape)
    body_quat_points_w = body_quat_w.unsqueeze(2).expand(*point_shape[:-1], 4)
    point_offsets_w = quat_apply(body_quat_points_w, point_offsets_b)
    point_pos_w = body_pos_w.unsqueeze(2) + point_offsets_w
    point_vel_w = body_link_lin_vel_w.unsqueeze(2) + torch.linalg.cross(
        body_ang_vel_w.unsqueeze(2).expand(point_shape), point_offsets_w, dim=-1
    )
    return point_pos_w, point_vel_w


class FeetTouchdownEdgeVelocityL2(ManagerTermBase):
    """Penalize the fastest descending sole corner near touchdown [(m/s)^2].

    Add the previous policy sample cost times pre_touchdown_scale once at a new
    landing, then evaluate unscaled current samples for a fixed window starting
    at first contact. Each foot takes the
    maximum squared downward speed of its four corners; foot costs are summed.
    Upward and horizontal translation are not penalized. Recontact inside the
    window neither restarts it nor repeats the pre-touchdown cost. The window
    persists through brief contact loss and rounds up to a policy-step boundary.
    Reward Manager supplies the weight and dt; this term applies neither.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize independent per-foot sample history and landing windows."""
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]
        asset_cfg = cfg.params["asset_cfg"]
        self._geometry = tuple(
            cfg.params.get(name, default)
            for name, default in (
                ("toe_x", 0.175), ("heel_x", -0.060), ("half_width", 0.045), ("sole_z", 0.0)
            )
        )
        toe_x, heel_x, half_width, sole_z = self._geometry
        if not all(math.isfinite(value) for value in self._geometry) or toe_x <= heel_x or half_width <= 0.0:
            raise ValueError("Invalid sole geometry: require finite offsets, toe_x > heel_x, and half_width > 0.")
        self._landing_window_s = cfg.params.get("landing_window_s", 0.10)
        self._pre_touchdown_scale = cfg.params.get("pre_touchdown_scale", 1.0)
        if not math.isfinite(self._pre_touchdown_scale) or self._pre_touchdown_scale < 0.0:
            raise ValueError("pre_touchdown_scale must be finite and non-negative.")
        self._step_dt = env.step_dt
        if not math.isfinite(self._step_dt) or self._step_dt <= 0.0:
            raise ValueError("Policy step_dt must be finite and positive.")
        if not math.isfinite(self._landing_window_s) or self._landing_window_s < self._step_dt:
            raise ValueError("landing_window_s must be finite and at least one policy step.")
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if not sensor.cfg.track_air_time:
            raise ValueError("FeetTouchdownEdgeVelocityL2 requires track_air_time=True.")
        num_feet = len(sensor_cfg.body_ids)
        if num_feet == 0 or num_feet != len(asset_cfg.body_ids):
            raise ValueError("Expected matching nonempty sensor and asset foot selections.")

        self._sole_points_b = torch.tensor(
            [
                [toe_x, half_width, sole_z],
                [toe_x, -half_width, sole_z],
                [heel_x, half_width, sole_z],
                [heel_x, -half_width, sole_z],
            ],
            device=env.device,
        )
        self._window_steps = max(1, math.ceil(self._landing_window_s / self._step_dt - 1.0e-9))
        self._early_steps = max(1, math.ceil(0.02 / self._step_dt - 1.0e-9))
        self._window_ms = round(self._window_steps * self._step_dt * 1000.0)
        shape = (env.num_envs, num_feet)
        self._steps_left = torch.zeros(shape, dtype=torch.long, device=env.device)
        self._previous_speed = torch.zeros(shape, device=env.device)
        self._previous_in_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self._has_previous_sample = torch.zeros_like(self._previous_in_contact)
        self._early_peak_speed = torch.zeros(shape, device=env.device)
        self._late_peak_speed = torch.zeros(shape, device=env.device)
        self._early_peak_sum = torch.zeros(env.num_envs, device=env.device)
        self._late_peak_sum = torch.zeros_like(self._early_peak_sum)
        self._completed_landings = torch.zeros_like(self._early_peak_sum)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        landing_window_s: float = 0.10,
        toe_x: float = 0.175,
        heel_x: float = -0.060,
        half_width: float = 0.045,
        sole_z: float = 0.0,
        pre_touchdown_scale: float = 1.0,
    ) -> torch.Tensor:
        """Return raw edge cost and advance fixed landing windows.

        Args:
            env: Environment supplying foot states and the policy timestep.
            sensor_cfg: Ordered foot contact selection, matching asset_cfg.
            asset_cfg: Ordered foot-body selection.
            landing_window_s: Post-contact evaluation duration [s].
            toe_x: Front sole offset from the foot-link origin [m].
            heel_x: Rear sole offset from the foot-link origin [m].
            half_width: Sole lateral half-width [m].
            sole_z: Sole height in the foot-link frame [m].
            pre_touchdown_scale: Multiplier on the previous sample's squared
                speed cost, applied once per landing. Does not scale logged speeds.

        Returns:
            Summed foot penalties [(m/s)^2], shape (num_envs,).
        """
        if (
            landing_window_s != self._landing_window_s
            or env.step_dt != self._step_dt
            or (toe_x, heel_x, half_width, sole_z) != self._geometry
        ):
            raise ValueError("Landing window, policy timestep, and sole geometry cannot change after initialization.")
        if pre_touchdown_scale != self._pre_touchdown_scale:
            raise ValueError("pre_touchdown_scale cannot change after initialization.")
        asset = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        _, point_vel_w = _sole_point_kinematics(asset, asset_cfg, self._sole_points_b)
        speed = torch.relu(-point_vel_w[..., 2]).amax(dim=-1)
        in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        if (
            speed.shape != self._steps_left.shape
            or in_contact.shape != speed.shape
            or first_contact.shape != speed.shape
        ):
            raise ValueError("Foot velocity/contact shapes do not match stored landing state.")

        new_landing = first_contact & self._has_previous_sample & (~self._previous_in_contact)
        new_landing &= self._steps_left == 0
        self._steps_left = torch.where(new_landing, self._window_steps, self._steps_left)
        active = self._steps_left > 0
        elapsed_steps = self._window_steps - self._steps_left
        penalty = torch.sum(
            speed.square() * active + pre_touchdown_scale * self._previous_speed.square() * new_landing,
            dim=1,
        )

        early = active & (elapsed_steps < self._early_steps)
        late = active & (~early)
        self._early_peak_speed = torch.where(
            early, torch.maximum(self._early_peak_speed, speed), self._early_peak_speed
        )
        self._late_peak_speed = torch.where(
            late, torch.maximum(self._late_peak_speed, speed), self._late_peak_speed
        )
        self._steps_left = (self._steps_left - 1).clamp_min(0)
        complete = active & (self._steps_left == 0)
        self._early_peak_sum += torch.sum(self._early_peak_speed * complete, dim=1)
        self._late_peak_sum += torch.sum(self._late_peak_speed * complete, dim=1)
        self._completed_landings += torch.sum(complete, dim=1)
        self._early_peak_speed.masked_fill_(complete, 0.0)
        self._late_peak_speed.masked_fill_(complete, 0.0)
        self._previous_speed.copy_(speed)
        self._previous_in_contact.copy_(in_contact)
        self._has_previous_sample.fill_(True)
        return penalty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log completed-window peak speeds [m/s] and clear selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        for name, values in (
            ("mean_peak_edge_downward_speed_0_20ms", self._early_peak_sum),
            (f"mean_peak_edge_downward_speed_20_{self._window_ms}ms", self._late_peak_sum),
        ):
            _log_episode_event_average(
                self._env, f"Metrics/feet_touchdown/{name}", values, self._completed_landings, env_ids
            )
        for values in (
            self._steps_left, self._previous_speed, self._previous_in_contact, self._has_previous_sample,
            self._early_peak_speed, self._late_peak_speed, self._early_peak_sum, self._late_peak_sum,
            self._completed_landings,
        ):
            values[env_ids] = 0


class FeetTouchdownDiagnostics(ManagerTermBase):
    """Record sole-edge approach speed and split touchdown normal-force peaks.

    The diagnostic reconstructs the four corners of each rectangular sole from
    the foot-body link-origin pose and velocity. Point velocities include the
    rigid-body angular contribution ``omega x r``, with ``r`` measured from that
    same link origin. It reports toe, heel, and lowest-corner
    approach speeds at first contact, then separates the normal-force peak into
    early-impact and subsequent weight-acceptance windows. The returned tensor
    is always zero, so this term never contributes to the policy reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize sole geometry and per-environment touchdown statistics."""
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        toe_x = cfg.params.get("toe_x", 0.175)
        heel_x = cfg.params.get("heel_x", -0.060)
        half_width = cfg.params.get("half_width", 0.045)
        sole_z = cfg.params.get("sole_z", 0.0)
        early_window_s = cfg.params.get("early_window_s", 0.02)
        late_window_s = cfg.params.get("late_window_s", 0.10)
        self._validate_parameters(toe_x, heel_x, half_width, early_window_s, late_window_s)

        num_sensor_bodies = len(sensor_cfg.body_ids)
        num_asset_bodies = len(asset_cfg.body_ids)
        if num_sensor_bodies != num_asset_bodies:
            raise ValueError(
                f"Selected {num_asset_bodies} foot bodies but received "
                f"{num_sensor_bodies} contact bodies."
            )

        self._sole_points_b = torch.tensor(
            [
                [toe_x, half_width, sole_z],
                [toe_x, -half_width, sole_z],
                [heel_x, half_width, sole_z],
                [heel_x, -half_width, sole_z],
            ],
            device=env.device,
        )
        self._early_window_s = early_window_s
        self._late_window_s = late_window_s
        self._early_window_ms = round(1000.0 * early_window_s)
        self._late_window_ms = round(1000.0 * late_window_s)

        event_shape = (env.num_envs, num_asset_bodies)
        point_shape = (*event_shape, 4)
        self._previous_point_vel_w = torch.zeros((*point_shape, 3), device=env.device)
        self._previous_point_height_w = torch.zeros(point_shape, device=env.device)
        self._previous_in_contact = torch.zeros(event_shape, dtype=torch.bool, device=env.device)
        self._has_previous_sample = torch.zeros(event_shape, dtype=torch.bool, device=env.device)

        self._toe_abs_vx_sum = torch.zeros(env.num_envs, device=env.device)
        self._toe_abs_vy_sum = torch.zeros(env.num_envs, device=env.device)
        self._toe_downward_speed_sum = torch.zeros(env.num_envs, device=env.device)
        self._heel_abs_vx_sum = torch.zeros(env.num_envs, device=env.device)
        self._heel_abs_vy_sum = torch.zeros(env.num_envs, device=env.device)
        self._heel_downward_speed_sum = torch.zeros(env.num_envs, device=env.device)
        self._lower_edge_planar_speed_sum = torch.zeros(env.num_envs, device=env.device)
        self._lower_edge_downward_speed_sum = torch.zeros(env.num_envs, device=env.device)
        self._point_touchdown_count = torch.zeros(env.num_envs, device=env.device)

        self._landing_active = torch.zeros(event_shape, dtype=torch.bool, device=env.device)
        self._landing_elapsed = torch.zeros(event_shape, device=env.device)
        self._early_peak_normal_force = torch.zeros(event_shape, device=env.device)
        self._late_peak_normal_force = torch.zeros(event_shape, device=env.device)
        self._early_peak_normal_force_sum = torch.zeros(env.num_envs, device=env.device)
        self._late_peak_normal_force_sum = torch.zeros(env.num_envs, device=env.device)
        self._force_touchdown_count = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        toe_x: float = 0.175,
        heel_x: float = -0.060,
        half_width: float = 0.045,
        sole_z: float = 0.0,
        early_window_s: float = 0.02,
        late_window_s: float = 0.10,
    ) -> torch.Tensor:
        """Update touchdown diagnostics and return zero reward."""
        self._validate_parameters(toe_x, heel_x, half_width, early_window_s, late_window_s)
        if early_window_s != self._early_window_s or late_window_s != self._late_window_s:
            raise ValueError("Touchdown diagnostic windows cannot change after initialization.")

        asset = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        point_pos_w, point_vel_w = self._sole_point_kinematics(asset, asset_cfg)
        in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

        expected_point_shape = self._previous_point_height_w.shape
        if point_pos_w.shape[:-1] != expected_point_shape or point_vel_w.shape != self._previous_point_vel_w.shape:
            raise ValueError(
                f"Sole point shapes {point_pos_w.shape}/{point_vel_w.shape} do not match "
                f"stored shapes {expected_point_shape}/{self._previous_point_vel_w.shape}."
            )

        valid_touchdown = first_contact & self._has_previous_sample & (~self._previous_in_contact)
        self._accumulate_point_metrics(valid_touchdown)
        self._accumulate_force_metrics(env, contact_sensor, sensor_cfg, in_contact, first_contact)

        self._previous_point_vel_w.copy_(point_vel_w)
        self._previous_point_height_w.copy_(point_pos_w[..., 2])
        self._previous_in_contact.copy_(in_contact)
        self._has_previous_sample.fill_(True)
        return torch.zeros(env.num_envs, device=point_vel_w.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log episode touchdown diagnostics and clear selected environments."""
        if env_ids is None:
            env_ids = slice(None)

        point_metrics = {
            "mean_pre_touchdown_toe_abs_vx": self._toe_abs_vx_sum,
            "mean_pre_touchdown_toe_abs_vy": self._toe_abs_vy_sum,
            "mean_pre_touchdown_toe_downward_speed": self._toe_downward_speed_sum,
            "mean_pre_touchdown_heel_abs_vx": self._heel_abs_vx_sum,
            "mean_pre_touchdown_heel_abs_vy": self._heel_abs_vy_sum,
            "mean_pre_touchdown_heel_downward_speed": self._heel_downward_speed_sum,
            "mean_pre_touchdown_lower_edge_planar_speed": self._lower_edge_planar_speed_sum,
            "mean_pre_touchdown_lower_edge_downward_speed": self._lower_edge_downward_speed_sum,
        }
        for name, value_sum in point_metrics.items():
            _log_episode_event_average(
                self._env,
                f"Metrics/feet_touchdown/{name}",
                value_sum,
                self._point_touchdown_count,
                env_ids,
            )

        force_metrics = {
            f"mean_peak_normal_force_0_{self._early_window_ms}ms": self._early_peak_normal_force_sum,
            (
                f"mean_peak_normal_force_{self._early_window_ms}_{self._late_window_ms}ms"
            ): self._late_peak_normal_force_sum,
        }
        for name, value_sum in force_metrics.items():
            _log_episode_event_average(
                self._env,
                f"Metrics/feet_touchdown/{name}",
                value_sum,
                self._force_touchdown_count,
                env_ids,
            )

        for tensor in (
            self._previous_point_vel_w,
            self._previous_point_height_w,
            self._toe_abs_vx_sum,
            self._toe_abs_vy_sum,
            self._toe_downward_speed_sum,
            self._heel_abs_vx_sum,
            self._heel_abs_vy_sum,
            self._heel_downward_speed_sum,
            self._lower_edge_planar_speed_sum,
            self._lower_edge_downward_speed_sum,
            self._point_touchdown_count,
            self._landing_elapsed,
            self._early_peak_normal_force,
            self._late_peak_normal_force,
            self._early_peak_normal_force_sum,
            self._late_peak_normal_force_sum,
            self._force_touchdown_count,
        ):
            tensor[env_ids] = 0.0
        self._previous_in_contact[env_ids] = False
        self._has_previous_sample[env_ids] = False
        self._landing_active[env_ids] = False

    def _sole_point_kinematics(self, asset, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sole-corner world positions [m] and velocities [m/s]."""
        return _sole_point_kinematics(asset, asset_cfg, self._sole_points_b)

    def _accumulate_point_metrics(self, valid_touchdown: torch.Tensor) -> None:
        """Accumulate pre-touchdown toe, heel, and lowest-corner velocities."""
        point_vel_w = self._previous_point_vel_w
        point_height_w = self._previous_point_height_w
        point_abs_vx = torch.abs(point_vel_w[..., 0])
        point_abs_vy = torch.abs(point_vel_w[..., 1])
        point_planar_speed = torch.linalg.vector_norm(point_vel_w[..., :2], dim=-1)
        point_downward_speed = torch.relu(-point_vel_w[..., 2])

        toe_index = torch.argmin(point_height_w[..., :2], dim=-1, keepdim=True)
        heel_index = torch.argmin(point_height_w[..., 2:], dim=-1, keepdim=True) + 2
        lower_edge_index = torch.argmin(point_height_w, dim=-1, keepdim=True)

        def select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, dim=2, index=indices).squeeze(2)

        metric_values = (
            (self._toe_abs_vx_sum, select(point_abs_vx, toe_index)),
            (self._toe_abs_vy_sum, select(point_abs_vy, toe_index)),
            (self._toe_downward_speed_sum, select(point_downward_speed, toe_index)),
            (self._heel_abs_vx_sum, select(point_abs_vx, heel_index)),
            (self._heel_abs_vy_sum, select(point_abs_vy, heel_index)),
            (self._heel_downward_speed_sum, select(point_downward_speed, heel_index)),
            (self._lower_edge_planar_speed_sum, select(point_planar_speed, lower_edge_index)),
            (self._lower_edge_downward_speed_sum, select(point_downward_speed, lower_edge_index)),
        )
        for value_sum, values in metric_values:
            value_sum += torch.sum(values * valid_touchdown, dim=1)
        self._point_touchdown_count += torch.sum(valid_touchdown, dim=1)

    def _accumulate_force_metrics(
        self,
        env: ManagerBasedRLEnv,
        contact_sensor: ContactSensor,
        sensor_cfg: SceneEntityCfg,
        in_contact: torch.Tensor,
        first_contact: torch.Tensor,
    ) -> None:
        """Accumulate normal-force peaks in early and late landing windows [N]."""
        force_history = contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
        ground_force_history = torch.nan_to_num(force_history).sum(dim=3)
        peak_normal_force = torch.relu(ground_force_history[..., 2]).max(dim=1).values
        if peak_normal_force.shape != self._landing_active.shape:
            raise ValueError(
                f"Peak normal-force shape {peak_normal_force.shape} does not match "
                f"landing state shape {self._landing_active.shape}."
            )

        self._landing_active |= first_contact
        self._landing_elapsed = torch.where(
            first_contact,
            torch.zeros_like(self._landing_elapsed),
            self._landing_elapsed,
        )
        self._early_peak_normal_force = torch.where(
            first_contact,
            torch.zeros_like(self._early_peak_normal_force),
            self._early_peak_normal_force,
        )
        self._late_peak_normal_force = torch.where(
            first_contact,
            torch.zeros_like(self._late_peak_normal_force),
            self._late_peak_normal_force,
        )

        early_active = self._landing_active & (self._landing_elapsed < self._early_window_s)
        late_active = (
            self._landing_active
            & (self._landing_elapsed >= self._early_window_s)
            & (self._landing_elapsed < self._late_window_s)
        )
        self._early_peak_normal_force = torch.where(
            early_active,
            torch.maximum(self._early_peak_normal_force, peak_normal_force),
            self._early_peak_normal_force,
        )
        self._late_peak_normal_force = torch.where(
            late_active,
            torch.maximum(self._late_peak_normal_force, peak_normal_force),
            self._late_peak_normal_force,
        )

        next_elapsed = self._landing_elapsed + env.step_dt * self._landing_active
        time_tolerance = max(1.0e-9, env.step_dt * 1.0e-4)
        landing_complete = self._landing_active & (
            (next_elapsed >= self._late_window_s - time_tolerance)
            | ((~in_contact) & (~first_contact))
        )
        self._early_peak_normal_force_sum += torch.sum(
            self._early_peak_normal_force * landing_complete,
            dim=1,
        )
        self._late_peak_normal_force_sum += torch.sum(
            self._late_peak_normal_force * landing_complete,
            dim=1,
        )
        self._force_touchdown_count += torch.sum(landing_complete, dim=1)

        self._landing_elapsed = torch.where(
            landing_complete,
            torch.zeros_like(next_elapsed),
            next_elapsed,
        )
        self._early_peak_normal_force = torch.where(
            landing_complete,
            torch.zeros_like(self._early_peak_normal_force),
            self._early_peak_normal_force,
        )
        self._late_peak_normal_force = torch.where(
            landing_complete,
            torch.zeros_like(self._late_peak_normal_force),
            self._late_peak_normal_force,
        )
        self._landing_active &= ~landing_complete

    @staticmethod
    def _validate_parameters(
        toe_x: float,
        heel_x: float,
        half_width: float,
        early_window_s: float,
        late_window_s: float,
    ) -> None:
        """Validate sole geometry [m] and landing windows [s]."""
        if toe_x <= heel_x:
            raise ValueError(f"toe_x must exceed heel_x, received {toe_x} and {heel_x}.")
        if half_width <= 0.0:
            raise ValueError(f"half_width must be positive, received {half_width}.")
        if early_window_s <= 0.0 or late_window_s <= early_window_s:
            raise ValueError(
                "Landing windows must satisfy 0 < early_window_s < late_window_s, received "
                f"{early_window_s} and {late_window_s}."
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
    """Track the peak force in an initial landing window and optionally penalize it [N].

    Setting :paramref:`metric_only` to ``True`` preserves the episode metric
    while returning an identically zero reward value.
    """

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
        metric_only: bool = False,
    ) -> torch.Tensor:
        """Complete each landing window, record its peak, and return its optional penalty."""
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

        self._touchdown_force_sum += torch.sum(self._landing_peak_force * landing_complete, dim=1)
        self._touchdown_count += torch.sum(landing_complete, dim=1)

        if metric_only:
            penalty = torch.zeros(env.num_envs, device=self._landing_peak_force.device)
        else:
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
    yaw_lift_fraction: float = 0.5,
) -> torch.Tensor:
    """Reward command-directed swing feet near a target flat-ground clearance.

    The height is the selected foot-body origin's world-Z coordinate relative
    to the environment origin. For linear commands, the velocity gate uses only
    foot progress along the commanded planar direction. For pure-yaw commands,
    :paramref:`yaw_lift_fraction` rewards vertical clearance without requiring
    planar motion, while the remaining fraction rewards progress along each
    foot's commanded tangent about the root. The term is disabled when there is
    no planar or yaw command and in standing environments.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive, received {std}.")
    if velocity_scale <= 0.0:
        raise ValueError(f"velocity_scale must be positive, received {velocity_scale}.")
    if not 0.0 <= yaw_lift_fraction <= 1.0:
        raise ValueError(
            f"yaw_lift_fraction must be within [0, 1], received {yaw_lift_fraction}."
        )

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

    command = env.command_manager.get_command(command_name)
    command_xy = command[:, :2]
    command_xy_norm = torch.linalg.vector_norm(command_xy, dim=-1)
    eps = torch.finfo(command.dtype).eps
    command_xy_dir = command_xy / command_xy_norm.clamp_min(eps).unsqueeze(-1)
    foot_progress_speed = torch.relu(
        torch.sum(foot_vel_yaw[..., :2] * command_xy_dir.unsqueeze(1), dim=-1)
    )
    linear_gate = torch.tanh(foot_progress_speed / velocity_scale)

    foot_pos_root_w = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)
    foot_pos_root_yaw = quat_apply_inverse(root_yaw_w, foot_pos_root_w)
    yaw_tangent_xy = torch.stack(
        (-foot_pos_root_yaw[..., 1], foot_pos_root_yaw[..., 0]),
        dim=-1,
    )
    yaw_tangent_xy *= torch.sign(command[:, 2]).view(-1, 1, 1)
    yaw_tangent_dir = yaw_tangent_xy / torch.linalg.vector_norm(
        yaw_tangent_xy,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    foot_vel_root_yaw = quat_apply_inverse(
        root_yaw_w,
        foot_vel_w - asset.data.root_lin_vel_w.unsqueeze(1),
    )
    yaw_progress_speed = torch.relu(
        torch.sum(foot_vel_root_yaw[..., :2] * yaw_tangent_dir, dim=-1)
    )
    yaw_progress_gate = torch.tanh(yaw_progress_speed / velocity_scale)
    yaw_gate = yaw_lift_fraction + (1.0 - yaw_lift_fraction) * yaw_progress_gate

    linear_command = command_xy_norm > eps
    yaw_command = (~linear_command) & (torch.abs(command[:, 2]) > eps)
    velocity_gate = torch.zeros_like(foot_progress_speed)
    velocity_gate = torch.where(linear_command.unsqueeze(-1), linear_gate, velocity_gate)
    velocity_gate = torch.where(yaw_command.unsqueeze(-1), yaw_gate, velocity_gate)
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


class FeetTouchdownPitchL2(ManagerTermBase):
    """Penalize sagittal sole tilt once per observed airborne-to-contact transition.

    Use the same dimensionless yaw-removed sole-normal X component squared as
    :func:`feet_swing_pitch_l2`. At first contact, take the larger error from the
    preceding policy sample and the current sample, then sum selected feet.
    Continued stance and toe-off incur no cost from this term. Both toe-up and
    toe-down are penalized; this does not prescribe heel-first contact.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize per-foot error, contact and sample-validity histories."""
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]
        asset_cfg = cfg.params["asset_cfg"]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        if not sensor.cfg.track_air_time:
            raise ValueError("FeetTouchdownPitchL2 requires track_air_time=True.")
        num_feet = len(sensor_cfg.body_ids)
        if num_feet == 0 or num_feet != len(asset_cfg.body_ids):
            raise ValueError("Expected matching nonempty sensor and asset foot selections.")
        shape = (env.num_envs, num_feet)
        self._previous_pitch_error = torch.zeros(shape, device=env.device)
        self._previous_in_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self._has_previous_sample = torch.zeros_like(self._previous_in_contact)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the unweighted, dimensionless touchdown-pitch cost.

        Args:
            env: Environment supplying foot poses and the policy timestep [s].
            asset_cfg: Ordered foot-body selection for world-frame orientations.
            sensor_cfg: Matching ordered foot-contact selection.

        Returns:
            Summed per-foot squared normal components, shape (num_envs,).
            Reward Manager applies the configured weight and timestep [s].
        """
        asset = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
        foot_up_local = torch.zeros_like(foot_quat_w[..., :3])
        foot_up_local[..., 2] = 1.0
        foot_up_w = quat_apply(foot_quat_w, foot_up_local)
        foot_up_yaw = quat_apply_inverse(yaw_quat(foot_quat_w), foot_up_w)
        pitch_error = foot_up_yaw[..., 0].square()
        in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
        first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        if (
            pitch_error.shape != self._previous_pitch_error.shape
            or in_contact.shape != pitch_error.shape
            or first_contact.shape != pitch_error.shape
        ):
            raise ValueError("Foot orientation/contact shapes do not match stored touchdown-pitch state.")

        valid_touchdown = first_contact & self._has_previous_sample & (~self._previous_in_contact)
        penalty = torch.sum(
            torch.maximum(self._previous_pitch_error, pitch_error) * valid_touchdown, dim=1
        )
        self._previous_pitch_error.copy_(pitch_error)
        self._previous_in_contact.copy_(in_contact)
        self._has_previous_sample.fill_(True)
        return penalty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear selected histories so initial contacts are not treated as landings."""
        if env_ids is None:
            env_ids = slice(None)
        self._previous_pitch_error[env_ids] = 0.0
        self._previous_in_contact[env_ids] = False
        self._has_previous_sample[env_ids] = False


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
