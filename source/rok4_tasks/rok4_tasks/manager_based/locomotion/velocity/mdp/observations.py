"""Actuator-space observations for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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


def base_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root height above the flat environment origin [m]."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def foot_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Selected foot-body origin heights above the flat environment origin [m]."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    return foot_pos_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(-1)


def foot_contact_flag(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Binary contact state for the selected feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    return (contact_time > 0.0).to(dtype=contact_time.dtype)


def foot_current_air_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Current uninterrupted air time for the selected feet [s]."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]


def _adapt_actuator(asset: Articulation, actuator_name: str) -> RoK4AdaptActuator:
    """Return the configured RoK4 ADAPT actuator with type validation."""
    actuator = asset.actuators.get(actuator_name)
    if not isinstance(actuator, RoK4AdaptActuator):
        raise TypeError(
            f"Asset actuator '{actuator_name}' must be RoK4AdaptActuator, received {type(actuator).__name__}."
        )
    return actuator
