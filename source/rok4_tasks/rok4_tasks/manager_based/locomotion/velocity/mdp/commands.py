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
    """Uniform velocity command with periodic all-environment stop windows."""

    cfg: RoK4PeriodicFreezeVelocityCommandCfg

    def __init__(self, cfg: RoK4PeriodicFreezeVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        duration_min, duration_max = cfg.periodic_freeze_duration_range_s
        if cfg.periodic_freeze_interval_s <= 0.0:
            raise ValueError("periodic_freeze_interval_s must be positive.")
        if duration_min <= 0.0 or duration_max < duration_min:
            raise ValueError(
                "periodic_freeze_duration_range_s must contain positive values in ascending order."
            )
        if duration_max >= cfg.periodic_freeze_interval_s:
            raise ValueError("Periodic freeze duration must be shorter than its interval.")

        self._periodic_freeze_active = False
        self._periodic_freeze_elapsed_s = 0.0
        self._periodic_freeze_time_left_s = 0.0
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = super().__str__()
        msg += f"\n\tPeriodic freeze enabled: {self.cfg.periodic_freeze_enabled}"
        if self.cfg.periodic_freeze_enabled:
            msg += f"\n\tPeriodic freeze interval: {self.cfg.periodic_freeze_interval_s} s"
            msg += f"\n\tPeriodic freeze duration: {self.cfg.periodic_freeze_duration_range_s} s"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset commands while preserving an active global freeze window."""
        reset_env_ids = slice(None) if env_ids is None else env_ids
        extras = super().reset(env_ids)
        if self.cfg.periodic_freeze_enabled and self._periodic_freeze_active:
            self._enforce_periodic_freeze(reset_env_ids)
        return extras

    def compute(self, dt: float):
        """Compute commands and apply the periodic all-environment freeze schedule."""
        super().compute(dt)
        if not self.cfg.periodic_freeze_enabled:
            return

        self._periodic_freeze_elapsed_s += dt
        if self._periodic_freeze_active:
            self._periodic_freeze_time_left_s -= dt
            if self._periodic_freeze_time_left_s <= 0.0:
                self._periodic_freeze_active = False
                self._resample(self._all_env_ids)
                self._update_command()
            else:
                self._enforce_periodic_freeze(self._all_env_ids)
        elif self._periodic_freeze_elapsed_s >= self.cfg.periodic_freeze_interval_s:
            self._periodic_freeze_active = True
            self._periodic_freeze_elapsed_s = 0.0
            self._periodic_freeze_time_left_s = self._sample_periodic_freeze_duration()
            self._enforce_periodic_freeze(self._all_env_ids)

    def _sample_periodic_freeze_duration(self) -> float:
        """Sample one global command-freeze duration [s]."""
        return (
            torch.empty((), device=self.device)
            .uniform_(*self.cfg.periodic_freeze_duration_range_s)
            .item()
        )

    def _enforce_periodic_freeze(self, env_ids: Sequence[int] | slice):
        """Set selected environments to exact-zero standing commands."""
        self.is_standing_env[env_ids] = True
        self.vel_command_b[env_ids] = 0.0


@configclass
class RoK4PeriodicFreezeVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`RoK4PeriodicFreezeVelocityCommand`."""

    class_type: type = RoK4PeriodicFreezeVelocityCommand

    periodic_freeze_enabled: bool = True
    """Whether to periodically replace every environment command with an exact-zero command."""

    periodic_freeze_interval_s: float = 10.0
    """Time between the starts of consecutive global command-freeze windows [s]."""

    periodic_freeze_duration_range_s: tuple[float, float] = (1.5, 3.0)
    """Uniform range of global command-freeze durations [s]."""
