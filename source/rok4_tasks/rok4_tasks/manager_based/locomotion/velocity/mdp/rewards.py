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


def stand_still_joint_deviation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize squared joint-position deviations in designated standing environments [rad^2]."""
    asset = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    joint_pos_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(joint_pos_error), dim=1) * command_term.is_standing_env


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
    """Compute the weighted squared difference of clipped raw policy actions."""
    return _weighted_l2(action_delta)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order clipped raw-action changes."""
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
    """Penalize second-order clipped raw-action changes."""

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
