"""Command terms for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RoK4PeriodicFreezeVelocityCommand(UniformVelocityCommand):
    """Uniform velocity commands with episode roles and asynchronous stop windows."""

    cfg: RoK4PeriodicFreezeVelocityCommandCfg

    def __init__(self, cfg: RoK4PeriodicFreezeVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        if cfg.periodic_freeze_enabled:
            duration_min, duration_max = cfg.periodic_freeze_duration_range_s
            role_ratio_sum = cfg.mixed_env_ratio + cfg.standing_env_ratio + cfg.walking_env_ratio
            if min(cfg.mixed_env_ratio, cfg.standing_env_ratio, cfg.walking_env_ratio) < 0.0:
                raise ValueError("Command environment ratios must be non-negative.")
            if abs(role_ratio_sum - 1.0) > 1.0e-6:
                raise ValueError(f"Command environment ratios must sum to 1.0, received {role_ratio_sum}.")
            if cfg.periodic_freeze_interval_s <= 0.0:
                raise ValueError("periodic_freeze_interval_s must be positive.")
            if duration_min <= 0.0 or duration_max < duration_min:
                raise ValueError(
                    "periodic_freeze_duration_range_s must contain positive values in ascending order."
                )
            if duration_max >= cfg.periodic_freeze_interval_s:
                raise ValueError("Periodic freeze duration must be shorter than its interval.")
            if cfg.always_walking_min_lin_vel < 0.0:
                raise ValueError("always_walking_min_lin_vel must be non-negative.")

            max_lin_vel = torch.linalg.vector_norm(
                torch.tensor(
                    [
                        max(abs(value) for value in cfg.ranges.lin_vel_x),
                        max(abs(value) for value in cfg.ranges.lin_vel_y),
                    ]
                )
            ).item()
            if cfg.always_walking_min_lin_vel > max_lin_vel:
                raise ValueError(
                    "always_walking_min_lin_vel exceeds the largest command allowed by the configured x/y ranges."
                )

        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._is_mixed_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._is_always_standing_env = torch.zeros_like(self._is_mixed_env)
        self._is_always_walking_env = torch.zeros_like(self._is_mixed_env)
        self._mixed_freeze_active = torch.zeros_like(self._is_mixed_env)
        self._mixed_freeze_phase_s = torch.zeros(self.num_envs, device=self.device)
        self._mixed_freeze_duration_s = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = super().__str__()
        msg += f"\n\tPeriodic freeze enabled: {self.cfg.periodic_freeze_enabled}"
        if self.cfg.periodic_freeze_enabled:
            msg += (
                "\n\tEnvironment role ratios (mixed/standing/walking): "
                f"{self.cfg.mixed_env_ratio}/{self.cfg.standing_env_ratio}/{self.cfg.walking_env_ratio}"
            )
            msg += f"\n\tAsynchronous freeze interval: {self.cfg.periodic_freeze_interval_s} s"
            msg += f"\n\tAsynchronous freeze duration: {self.cfg.periodic_freeze_duration_range_s} s"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Assign episode roles and reset each mixed environment at a random schedule phase."""
        reset_env_ids = self._resolve_env_ids(env_ids)
        if self.cfg.periodic_freeze_enabled:
            self._sample_environment_roles(reset_env_ids)
            self._reset_mixed_freeze_schedule(reset_env_ids)
        return super().reset(reset_env_ids)

    def compute(self, dt: float):
        """Compute commands and advance each mixed environment's freeze schedule."""
        if not self.cfg.periodic_freeze_enabled:
            super().compute(dt)
            return

        previous_freeze_active = self._mixed_freeze_active.clone()
        self._mixed_freeze_phase_s[self._is_mixed_env] += dt
        wrapped_env_ids = (
            self._is_mixed_env & (self._mixed_freeze_phase_s >= self.cfg.periodic_freeze_interval_s)
        ).nonzero(as_tuple=False).flatten()
        if len(wrapped_env_ids) > 0:
            self._mixed_freeze_phase_s[wrapped_env_ids] = torch.remainder(
                self._mixed_freeze_phase_s[wrapped_env_ids], self.cfg.periodic_freeze_interval_s
            )
            self._sample_mixed_freeze_duration(wrapped_env_ids)

        self._mixed_freeze_active = self._is_mixed_env & (
            self._mixed_freeze_phase_s < self._mixed_freeze_duration_s
        )
        exited_freeze_env_ids = (previous_freeze_active & ~self._mixed_freeze_active).nonzero(
            as_tuple=False
        ).flatten()
        if len(exited_freeze_env_ids) > 0:
            self._resample(exited_freeze_env_ids)

        super().compute(dt)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands and enforce the selected episode role."""
        env_ids = self._resolve_env_ids(env_ids)
        super()._resample_command(env_ids)
        if not self.cfg.periodic_freeze_enabled:
            return

        walking_env_ids = env_ids[self._is_always_walking_env[env_ids]]
        self._resample_always_walking_command(walking_env_ids)

        self._apply_standing_mask(env_ids)

    def _update_command(self):
        """Apply role-derived standing masks before the parent post-processing."""
        if self.cfg.periodic_freeze_enabled:
            self._apply_standing_mask(self._all_env_ids)
        super()._update_command()

    def _sample_environment_roles(self, env_ids: torch.Tensor):
        """Sample mixed, always-standing, and always-walking roles for one episode."""
        role_sample = torch.rand(len(env_ids), device=self.device)
        mixed_upper = self.cfg.mixed_env_ratio
        standing_upper = mixed_upper + self.cfg.standing_env_ratio
        self._is_mixed_env[env_ids] = role_sample < mixed_upper
        self._is_always_standing_env[env_ids] = (role_sample >= mixed_upper) & (role_sample < standing_upper)
        self._is_always_walking_env[env_ids] = role_sample >= standing_upper

    def _reset_mixed_freeze_schedule(self, env_ids: torch.Tensor):
        """Randomize mixed-environment phases so stop windows are asynchronous."""
        self._mixed_freeze_phase_s[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            0.0, self.cfg.periodic_freeze_interval_s
        )
        self._sample_mixed_freeze_duration(env_ids)
        self._mixed_freeze_active[env_ids] = self._is_mixed_env[env_ids] & (
            self._mixed_freeze_phase_s[env_ids] < self._mixed_freeze_duration_s[env_ids]
        )

    def _sample_mixed_freeze_duration(self, env_ids: torch.Tensor):
        """Sample per-environment standing durations [s]."""
        self._mixed_freeze_duration_s[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.periodic_freeze_duration_range_s
        )

    def _resample_always_walking_command(self, env_ids: torch.Tensor):
        """Reject near-zero linear commands for always-walking environments."""
        if len(env_ids) == 0 or self.cfg.always_walking_min_lin_vel == 0.0:
            return

        pending_env_ids = env_ids[
            torch.linalg.vector_norm(self.vel_command_b[env_ids, :2], dim=1)
            < self.cfg.always_walking_min_lin_vel
        ]
        for _ in range(32):
            if len(pending_env_ids) == 0:
                return
            super()._resample_command(pending_env_ids)
            pending_env_ids = pending_env_ids[
                torch.linalg.vector_norm(self.vel_command_b[pending_env_ids, :2], dim=1)
                < self.cfg.always_walking_min_lin_vel
            ]
        raise RuntimeError("Failed to sample a non-zero command for always-walking environments.")

    def _apply_standing_mask(self, env_ids: torch.Tensor):
        """Synchronize exact-zero commands and the standing reward mask."""
        standing = self._is_always_standing_env[env_ids] | self._mixed_freeze_active[env_ids]
        self.is_standing_env[env_ids] = standing
        standing_env_ids = env_ids[standing]
        self.vel_command_b[standing_env_ids] = 0.0

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        """Convert supported environment index forms to a device tensor."""
        if env_ids is None:
            return self._all_env_ids
        if isinstance(env_ids, slice):
            return self._all_env_ids[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class RoK4PeriodicFreezeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`RoK4PeriodicFreezeVelocityCommand`."""

    class_type: type = RoK4PeriodicFreezeVelocityCommand

    periodic_freeze_enabled: bool = True
    """Whether to enable training-only episode roles and asynchronous standing windows."""

    mixed_env_ratio: float = 0.90
    """Probability that an environment alternates between moving and standing during an episode."""

    standing_env_ratio: float = 0.05
    """Probability that an environment remains standing throughout an episode."""

    walking_env_ratio: float = 0.05
    """Probability that an environment receives only moving commands throughout an episode."""

    periodic_freeze_interval_s: float = 10.0
    """Time between consecutive standing-window starts for each mixed environment [s]."""

    periodic_freeze_duration_range_s: tuple[float, float] = (1.5, 3.0)
    """Uniform range of per-environment standing durations [s]."""

    always_walking_min_lin_vel: float = 0.15
    """Minimum planar speed command accepted for always-walking environments [m/s]."""
