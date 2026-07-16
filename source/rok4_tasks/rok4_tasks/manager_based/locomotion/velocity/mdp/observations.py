"""Actuator-space observations for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from rok4_tasks.assets.robots.rok4_adapt import RoK4AdaptActuator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def actuator_pos_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Actuator positions relative to the default actuator pose [rad]."""
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return actuator.transmission.joint_to_actuator_position(joint_pos - default_joint_pos)


def actuator_vel_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "body",
) -> torch.Tensor:
    """Actuator velocities relative to the default actuator velocity [rad/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    actuator = _adapt_actuator(asset, actuator_name)
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    default_joint_vel = asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return actuator.transmission.joint_to_actuator_velocity(joint_vel - default_joint_vel)


def _adapt_actuator(asset: Articulation, actuator_name: str) -> RoK4AdaptActuator:
    """Return the configured RoK4 ADAPT actuator with type validation."""
    actuator = asset.actuators.get(actuator_name)
    if not isinstance(actuator, RoK4AdaptActuator):
        raise TypeError(
            f"Asset actuator '{actuator_name}' must be RoK4AdaptActuator, received {type(actuator).__name__}."
        )
    return actuator
