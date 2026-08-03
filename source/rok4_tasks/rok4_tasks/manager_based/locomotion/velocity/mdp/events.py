# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_correlated_material_buckets(
    static_friction_range: tuple[float, float],
    dynamic_friction_ratio: float,
    restitution_range: tuple[float, float],
    num_buckets: int,
) -> torch.Tensor:
    """Sample material buckets with a fixed dynamic-to-static friction ratio."""
    if static_friction_range[0] < 0.0 or static_friction_range[1] < static_friction_range[0]:
        raise ValueError("static_friction_range must contain non-negative values in ascending order.")
    if not 0.0 <= dynamic_friction_ratio <= 1.0:
        raise ValueError("dynamic_friction_ratio must be in [0, 1].")
    if restitution_range[0] < 0.0 or restitution_range[1] < restitution_range[0]:
        raise ValueError("restitution_range must contain non-negative values in ascending order.")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive.")

    static_friction = math_utils.sample_uniform(
        *static_friction_range,
        (num_buckets,),
        device="cpu",
    )
    dynamic_friction = static_friction * dynamic_friction_ratio
    restitution = math_utils.sample_uniform(*restitution_range, (num_buckets,), device="cpu")
    return torch.stack((static_friction, dynamic_friction, restitution), dim=-1)


class randomize_rigid_body_material_correlated(randomize_rigid_body_material):
    """Randomize one correlated contact material per environment.

    Every selected geometry in an environment receives the same material bucket. Dynamic friction is a fixed ratio
    of static friction, which preserves ``dynamic_friction <= static_friction`` and avoids left-right foot asymmetry.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        """Initialize correlated material buckets."""
        super().__init__(cfg, env)
        self.material_buckets = _sample_correlated_material_buckets(
            cfg.params["static_friction_range"],
            cfg.params["dynamic_friction_ratio"],
            cfg.params["restitution_range"],
            int(cfg.params["num_buckets"]),
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        default_static_friction: float,
        default_dynamic_friction: float,
        static_friction_range: tuple[float, float],
        dynamic_friction_ratio: float,
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Set the robot material baseline, then assign one Foot bucket per environment."""
        if default_static_friction < 0.0:
            raise ValueError("default_static_friction must be non-negative.")
        if not 0.0 <= default_dynamic_friction <= default_static_friction:
            raise ValueError("default_dynamic_friction must be in [0, default_static_friction].")

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        bucket_ids = torch.randint(0, num_buckets, (len(env_ids),), device="cpu")
        material_samples = self.material_buckets[bucket_ids]
        materials = self.asset.root_physx_view.get_material_properties()

        # All robot collision shapes use the same nominal G1/Digit-style friction unless overridden below.
        materials[env_ids, :, 0] = default_static_friction
        materials[env_ids, :, 1] = default_dynamic_friction

        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, None, :]
        else:
            materials[env_ids] = material_samples[:, None, :]

        self.asset.root_physx_view.set_material_properties(materials, env_ids)


def reset_joints_by_position_scale_and_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joint positions by default-pose scaling and sample absolute velocities.

    Args:
        env: Environment containing the articulation.
        env_ids: Environment indices to reset.
        position_range: Multiplicative range applied independently to each default joint position.
        velocity_range: Absolute joint velocity sampling range [rad/s].
        asset_cfg: Articulation and joint selection to reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids
    joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel[:] = math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
