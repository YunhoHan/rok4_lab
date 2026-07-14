"""Reward terms for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from rok4_tasks.assets.robots.rok4 import ROK4_ACTION_SCALE, ROK4_JOINT_ORDER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROK4_RELAXED_ACTION_IDS = (2, 3, 8, 9)
"""Action indices with reduced smoothness penalty, matching the Isaac Gym RoK4 setup."""

ROK4_ACTION_SCALE_VALUES = tuple(ROK4_ACTION_SCALE[joint_name] for joint_name in ROK4_JOINT_ORDER)
"""Action scale values ordered by :data:`ROK4_JOINT_ORDER`."""


def _weighted_l2(values: torch.Tensor) -> torch.Tensor:
    """Compute weighted squared sum."""
    weights = torch.ones_like(values)
    weights[:, list(ROK4_RELAXED_ACTION_IDS)] = 0.5
    return torch.sum(torch.square(values) * weights, dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied joint torques with the RoK4 actuator weights."""
    asset = env.scene[asset_cfg.name]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return _weighted_l2(applied_torque)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities with the RoK4 actuator weights."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return _weighted_l2(joint_vel)


def _scaled_action_l2(action_delta: torch.Tensor) -> torch.Tensor:
    """Compute weighted squared action-target offset difference."""
    action_scale = action_delta.new_tensor(ROK4_ACTION_SCALE_VALUES)
    return _weighted_l2(action_delta * action_scale)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order scaled action-target changes."""
    action_rate = env.action_manager.action - env.action_manager.prev_action
    penalty = _scaled_action_l2(action_rate)
    has_history = torch.any(env.action_manager.prev_action != 0.0, dim=1)
    return torch.where(has_history, penalty, torch.zeros_like(penalty))


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations using RoK4 joint weights."""
    asset = env.scene[asset_cfg.name]
    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return _weighted_l2(joint_acc)


def action_pos_limits(
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
    """Penalize second-order scaled action-target changes."""

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
        penalty = _scaled_action_l2(action_2nd_rate)

        penalty = torch.where(self._action_history_count >= 2, penalty, torch.zeros_like(penalty))
        self._prev_prev_action[:] = prev_action
        self._action_history_count[:] = torch.clamp(self._action_history_count + 1, max=2)
        return penalty
