# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _validate_range(name: str, value_range: tuple[float, float], *, positive: bool = False) -> None:
    """Validate an ascending scalar range."""
    lower, upper = value_range
    if upper < lower:
        raise ValueError(f"{name} must be in ascending order, received {value_range}.")
    if positive and lower <= 0.0:
        raise ValueError(f"{name} must contain positive values, received {value_range}.")


def _force_from_delta_velocity(
    delta_velocity_w: torch.Tensor,
    total_mass: torch.Tensor,
    duration_s: torch.Tensor,
) -> torch.Tensor:
    """Convert a desired world-frame velocity change into a constant force pulse [N]."""
    return delta_velocity_w * (total_mass / duration_s).unsqueeze(-1)


class RoK4MixedPush(ManagerTermBase):
    """Apply base-yaw-frame velocity pushes or impulse-equivalent force pulses.

    Each disturbance samples one planar velocity-change vector in the robot's base-yaw frame. A velocity event adds
    that vector directly to the root velocity. A force event instead applies a constant world-frame force
    ``F = m * delta_v / duration`` to the configured body for a finite duration. Exactly one mode is selected per
    event, so the disturbance frequency is not doubled.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize per-environment push and force-pulse timers."""
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset_cfg.body_ids, list) or len(self.asset_cfg.body_ids) != 1:
            raise ValueError("RoK4MixedPush requires exactly one configured body.")

        _validate_range("push_interval_range_s", cfg.params["push_interval_range_s"], positive=True)
        _validate_range("delta_velocity_x_range", cfg.params["delta_velocity_x_range"])
        _validate_range("delta_velocity_y_range", cfg.params["delta_velocity_y_range"])
        _validate_range("force_duration_range_s", cfg.params["force_duration_range_s"], positive=True)
        probability = float(cfg.params["velocity_push_probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("velocity_push_probability must be in [0, 1].")

        minimum_duration = cfg.params["force_duration_range_s"][0]
        if minimum_duration < env.step_dt:
            raise ValueError(
                "force_duration_range_s must be at least one policy step "
                f"({env.step_dt:.6f} s), received {cfg.params['force_duration_range_s']}."
            )

        self._push_time_left_s = torch.zeros(env.num_envs, device=env.device)
        self._force_time_left_s = torch.zeros_like(self._push_time_left_s)
        self._force_w = torch.zeros(env.num_envs, 3, device=env.device)
        self._total_mass: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset timers and clear active force pulses for selected environments."""
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self._push_time_left_s[env_ids] = self._sample_uniform(
            self.cfg.params["push_interval_range_s"], env_ids.numel()
        )
        self._force_time_left_s[env_ids] = 0.0
        self._force_w[env_ids] = 0.0
        self._write_force(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        push_interval_range_s: tuple[float, float],
        delta_velocity_x_range: tuple[float, float],
        delta_velocity_y_range: tuple[float, float],
        velocity_push_probability: float,
        force_duration_range_s: tuple[float, float],
    ) -> None:
        """Advance pulse timers and apply any disturbances due this policy step."""
        del asset_cfg
        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        dt = env.step_dt
        active_before = self._force_time_left_s[env_ids] > 0.0
        self._force_time_left_s[env_ids] = torch.clamp(self._force_time_left_s[env_ids] - dt, min=0.0)
        expired = active_before & (self._force_time_left_s[env_ids] <= 0.0)
        if torch.any(expired):
            expired_ids = env_ids[expired]
            self._force_w[expired_ids] = 0.0
            self._write_force(expired_ids)

        self._push_time_left_s[env_ids] -= dt
        due = self._push_time_left_s[env_ids] <= 0.0
        if not torch.any(due):
            return

        due_ids = env_ids[due]
        self._clear_active_force(due_ids)
        delta_velocity_b = torch.zeros(due_ids.numel(), 3, device=self.device)
        delta_velocity_b[:, 0] = self._sample_uniform(delta_velocity_x_range, due_ids.numel())
        delta_velocity_b[:, 1] = self._sample_uniform(delta_velocity_y_range, due_ids.numel())
        root_yaw_w = math_utils.yaw_quat(self.asset.data.root_quat_w[due_ids])
        delta_velocity_w = math_utils.quat_apply(root_yaw_w, delta_velocity_b)

        velocity_mask = torch.rand(due_ids.numel(), device=self.device) < velocity_push_probability
        if torch.any(velocity_mask):
            velocity_ids = due_ids[velocity_mask]
            root_velocity_w = self.asset.data.root_vel_w[velocity_ids].clone()
            root_velocity_w[:, :3] += delta_velocity_w[velocity_mask]
            self.asset.write_root_velocity_to_sim(root_velocity_w, env_ids=velocity_ids)

        force_mask = ~velocity_mask
        if torch.any(force_mask):
            force_ids = due_ids[force_mask]
            duration_s = self._sample_force_duration(force_duration_range_s, force_ids.numel(), dt)
            total_mass = self._get_total_mass()[force_ids]
            self._force_w[force_ids] = _force_from_delta_velocity(
                delta_velocity_w[force_mask], total_mass, duration_s
            )
            self._force_time_left_s[force_ids] = duration_s
            self._write_force(force_ids)

        self._push_time_left_s[due_ids] = self._sample_uniform(push_interval_range_s, due_ids.numel())

    def _resolve_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        """Return environment indices as a device tensor."""
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _sample_uniform(self, value_range: tuple[float, float], count: int) -> torch.Tensor:
        """Sample one scalar per environment from a uniform range."""
        return math_utils.sample_uniform(*value_range, (count,), self.device)

    def _sample_force_duration(
        self,
        duration_range_s: tuple[float, float],
        count: int,
        dt: float,
    ) -> torch.Tensor:
        """Sample force durations quantized to whole policy steps [s]."""
        minimum_steps = math.ceil(duration_range_s[0] / dt)
        maximum_steps = math.floor(duration_range_s[1] / dt)
        if maximum_steps < minimum_steps:
            maximum_steps = minimum_steps
        duration_steps = torch.randint(minimum_steps, maximum_steps + 1, (count,), device=self.device)
        return duration_steps.to(dtype=torch.float32) * dt

    def _get_total_mass(self) -> torch.Tensor:
        """Return cached post-startup-DR total robot mass [kg] for every environment."""
        if self._total_mass is None:
            masses = self.asset.root_physx_view.get_masses()
            self._total_mass = masses.sum(dim=1).to(device=self.device, dtype=torch.float32)
        return self._total_mass

    def _clear_active_force(self, env_ids: torch.Tensor) -> None:
        """Clear any force pulse before starting another disturbance."""
        active = self._force_time_left_s[env_ids] > 0.0
        if torch.any(active):
            active_ids = env_ids[active]
            self._force_time_left_s[active_ids] = 0.0
            self._force_w[active_ids] = 0.0
            self._write_force(active_ids)

    def _write_force(self, env_ids: torch.Tensor) -> None:
        """Write buffered world-frame base forces [N] for selected environments."""
        forces = self._force_w[env_ids, None, :]
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=self.asset_cfg.body_ids,
            env_ids=env_ids,
            is_global=True,
        )


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
