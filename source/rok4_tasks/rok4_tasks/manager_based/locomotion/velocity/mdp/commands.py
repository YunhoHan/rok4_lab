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

    _ROLE_MIXED = 0
    _ROLE_STANDING = 1
    _ROLE_WALKING = 2
    _ROLE_X = 3
    _ROLE_Y = 4
    _ROLE_YAW = 5
    _ROLE_X_YAW = 6
    _ROLE_NAMES = ("mixed", "standing", "walking", "x", "y", "yaw", "x_yaw")

    cfg: RoK4PeriodicFreezeVelocityCommandCfg

    def __init__(self, cfg: RoK4PeriodicFreezeVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        if cfg.periodic_freeze_enabled:
            duration_min, duration_max = cfg.periodic_freeze_duration_range_s
            role_ratios = self._role_ratios(cfg)
            role_ratio_sum = sum(role_ratios)
            if min(role_ratios) < 0.0:
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

            if cfg.x_env_ratio > 0.0 or cfg.x_yaw_env_ratio > 0.0:
                self._validate_dedicated_command_range(
                    "lin_vel_x", cfg.ranges.lin_vel_x, cfg.dedicated_x_min_abs_vel
                )
            if cfg.y_env_ratio > 0.0:
                self._validate_dedicated_command_range(
                    "lin_vel_y", cfg.ranges.lin_vel_y, cfg.dedicated_y_min_abs_vel
                )
            if cfg.yaw_env_ratio > 0.0 or cfg.x_yaw_env_ratio > 0.0:
                self._validate_dedicated_command_range(
                    "ang_vel_z", cfg.ranges.ang_vel_z, cfg.dedicated_yaw_min_abs_vel
                )

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
        self._role_boundaries = torch.cumsum(
            torch.tensor(self._role_ratios(cfg), device=self.device), dim=0
        )[:-1]
        self._environment_role = torch.full(
            (self.num_envs,), self._ROLE_MIXED, dtype=torch.long, device=self.device
        )
        self._is_periodic_freeze_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._periodic_freeze_active = torch.zeros_like(self._is_periodic_freeze_env)
        self._periodic_freeze_phase_s = torch.zeros(self.num_envs, device=self.device)
        self._periodic_freeze_duration_s = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = super().__str__()
        msg += f"\n\tPeriodic freeze enabled: {self.cfg.periodic_freeze_enabled}"
        if self.cfg.periodic_freeze_enabled:
            role_ratio_text = "/".join(f"{ratio:g}" for ratio in self._role_ratios(self.cfg))
            msg += f"\n\tEnvironment role ratios ({'/'.join(self._ROLE_NAMES)}): {role_ratio_text}"
            msg += f"\n\tAsynchronous freeze interval: {self.cfg.periodic_freeze_interval_s} s"
            msg += f"\n\tAsynchronous freeze duration: {self.cfg.periodic_freeze_duration_range_s} s"
        return msg

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Assign episode roles and reset each freeze-enabled environment at a random schedule phase."""
        reset_env_ids = self._resolve_env_ids(env_ids)
        if self.cfg.periodic_freeze_enabled:
            self._sample_environment_roles(reset_env_ids)
            self._reset_periodic_freeze_schedule(reset_env_ids)
        return super().reset(reset_env_ids)

    def compute(self, dt: float):
        """Compute commands and advance each eligible environment's freeze schedule."""
        if not self.cfg.periodic_freeze_enabled:
            super().compute(dt)
            return

        previous_freeze_active = self._periodic_freeze_active.clone()
        self._periodic_freeze_phase_s[self._is_periodic_freeze_env] += dt
        wrapped_env_ids = (
            self._is_periodic_freeze_env
            & (self._periodic_freeze_phase_s >= self.cfg.periodic_freeze_interval_s)
        ).nonzero(as_tuple=False).flatten()
        if len(wrapped_env_ids) > 0:
            self._periodic_freeze_phase_s[wrapped_env_ids] = torch.remainder(
                self._periodic_freeze_phase_s[wrapped_env_ids], self.cfg.periodic_freeze_interval_s
            )
            self._sample_periodic_freeze_duration(wrapped_env_ids)

        self._periodic_freeze_active = self._is_periodic_freeze_env & (
            self._periodic_freeze_phase_s < self._periodic_freeze_duration_s
        )
        exited_freeze_env_ids = (previous_freeze_active & ~self._periodic_freeze_active).nonzero(
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

        walking_env_ids = self._env_ids_for_role(env_ids, self._ROLE_WALKING)
        self._resample_always_walking_command(walking_env_ids)

        self._resample_x_command(self._env_ids_for_role(env_ids, self._ROLE_X))
        self._resample_y_command(self._env_ids_for_role(env_ids, self._ROLE_Y))
        self._resample_yaw_command(self._env_ids_for_role(env_ids, self._ROLE_YAW))
        self._resample_x_yaw_command(self._env_ids_for_role(env_ids, self._ROLE_X_YAW))

        self._apply_standing_mask(env_ids)

    def _update_command(self):
        """Apply role-derived standing masks before the parent post-processing."""
        if self.cfg.periodic_freeze_enabled:
            self._apply_standing_mask(self._all_env_ids)
        super()._update_command()

    def _sample_environment_roles(self, env_ids: torch.Tensor):
        """Sample one command role for each environment for the duration of its episode."""
        role_sample = torch.rand(len(env_ids), device=self.device)
        self._environment_role[env_ids] = torch.bucketize(role_sample, self._role_boundaries)
        roles = self._environment_role[env_ids]
        self._is_periodic_freeze_env[env_ids] = (roles != self._ROLE_STANDING) & (
            roles != self._ROLE_WALKING
        )

    def _reset_periodic_freeze_schedule(self, env_ids: torch.Tensor):
        """Randomize eligible-environment phases so stop windows are asynchronous."""
        self._periodic_freeze_phase_s[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            0.0, self.cfg.periodic_freeze_interval_s
        )
        self._sample_periodic_freeze_duration(env_ids)
        self._periodic_freeze_active[env_ids] = self._is_periodic_freeze_env[env_ids] & (
            self._periodic_freeze_phase_s[env_ids] < self._periodic_freeze_duration_s[env_ids]
        )

    def _sample_periodic_freeze_duration(self, env_ids: torch.Tensor):
        """Sample per-environment standing durations [s]."""
        self._periodic_freeze_duration_s[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
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

    def _resample_x_command(self, env_ids: torch.Tensor):
        """Sample balanced forward/backward commands with only ``vx`` active."""
        if len(env_ids) == 0:
            return
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = self._sample_balanced_signed_values(
            len(env_ids), self.cfg.ranges.lin_vel_x, self.cfg.dedicated_x_min_abs_vel
        )

    def _resample_y_command(self, env_ids: torch.Tensor):
        """Sample balanced left/right commands with only ``vy`` active."""
        if len(env_ids) == 0:
            return
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 1] = self._sample_balanced_signed_values(
            len(env_ids), self.cfg.ranges.lin_vel_y, self.cfg.dedicated_y_min_abs_vel
        )

    def _resample_yaw_command(self, env_ids: torch.Tensor):
        """Sample balanced clockwise/counter-clockwise commands with only ``wz`` active."""
        if len(env_ids) == 0:
            return
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 2] = self._sample_balanced_signed_values(
            len(env_ids), self.cfg.ranges.ang_vel_z, self.cfg.dedicated_yaw_min_abs_vel
        )

    def _resample_x_yaw_command(self, env_ids: torch.Tensor):
        """Sample balanced ``vx`` and ``wz`` commands while keeping ``vy`` zero."""
        if len(env_ids) == 0:
            return
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = self._sample_balanced_signed_values(
            len(env_ids), self.cfg.ranges.lin_vel_x, self.cfg.dedicated_x_min_abs_vel
        )
        self.vel_command_b[env_ids, 2] = self._sample_balanced_signed_values(
            len(env_ids), self.cfg.ranges.ang_vel_z, self.cfg.dedicated_yaw_min_abs_vel
        )

    def _sample_balanced_signed_values(
        self, count: int, value_range: tuple[float, float], minimum_abs_value: float
    ) -> torch.Tensor:
        """Sample either sign uniformly and sample its magnitude inside the configured range."""
        positive = torch.rand(count, device=self.device) < 0.5
        positive_values = torch.empty(count, device=self.device).uniform_(minimum_abs_value, value_range[1])
        negative_values = -torch.empty(count, device=self.device).uniform_(minimum_abs_value, abs(value_range[0]))
        return torch.where(positive, positive_values, negative_values)

    def _apply_standing_mask(self, env_ids: torch.Tensor):
        """Synchronize exact-zero commands and the standing reward mask."""
        standing = (self._environment_role[env_ids] == self._ROLE_STANDING) | self._periodic_freeze_active[env_ids]
        self.is_standing_env[env_ids] = standing
        standing_env_ids = env_ids[standing]
        self.vel_command_b[standing_env_ids] = 0.0

    def _env_ids_for_role(self, env_ids: torch.Tensor, role: int) -> torch.Tensor:
        """Return the requested subset of environment IDs for one episode role."""
        return env_ids[self._environment_role[env_ids] == role]

    @classmethod
    def _role_ratios(cls, cfg: RoK4PeriodicFreezeVelocityCommandCfg) -> tuple[float, ...]:
        """Return role ratios in the same order as :attr:`_ROLE_NAMES`."""
        return (
            cfg.mixed_env_ratio,
            cfg.standing_env_ratio,
            cfg.walking_env_ratio,
            cfg.x_env_ratio,
            cfg.y_env_ratio,
            cfg.yaw_env_ratio,
            cfg.x_yaw_env_ratio,
        )

    @staticmethod
    def _validate_dedicated_command_range(
        range_name: str, value_range: tuple[float, float], minimum_abs_value: float
    ):
        """Validate that a dedicated role can sample both signs away from zero."""
        if minimum_abs_value <= 0.0:
            raise ValueError(f"The minimum absolute value for {range_name} must be positive.")
        if value_range[0] > -minimum_abs_value or value_range[1] < minimum_abs_value:
            raise ValueError(
                f"The {range_name} range {value_range} must contain values <= {-minimum_abs_value} "
                f"and >= {minimum_abs_value} for balanced dedicated-role sampling."
            )

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

    mixed_env_ratio: float = 0.50
    """Probability that an environment alternates between moving and standing during an episode."""

    standing_env_ratio: float = 0.05
    """Probability that an environment remains standing throughout an episode."""

    walking_env_ratio: float = 0.05
    """Probability that an environment receives only moving commands throughout an episode."""

    x_env_ratio: float = 0.10
    """Probability of balanced forward/backward commands with only base-frame ``vx`` active."""

    y_env_ratio: float = 0.10
    """Probability of balanced left/right commands with only base-frame ``vy`` active."""

    yaw_env_ratio: float = 0.10
    """Probability of balanced turning commands with only base-frame ``wz`` active."""

    x_yaw_env_ratio: float = 0.10
    """Probability of balanced base-frame ``vx`` and ``wz`` commands with ``vy`` fixed to zero."""

    periodic_freeze_interval_s: float = 10.0
    """Time between consecutive standing-window starts for each freeze-enabled environment [s]."""

    periodic_freeze_duration_range_s: tuple[float, float] = (1.5, 3.0)
    """Uniform range of per-environment standing durations [s]."""

    always_walking_min_lin_vel: float = 0.15
    """Minimum planar speed command accepted for always-walking environments [m/s]."""

    dedicated_x_min_abs_vel: float = 0.15
    """Minimum absolute ``vx`` sampled by the ``x`` and ``x_yaw`` roles [m/s]."""

    dedicated_y_min_abs_vel: float = 0.15
    """Minimum absolute ``vy`` sampled by the ``y`` role [m/s]."""

    dedicated_yaw_min_abs_vel: float = 0.15
    """Minimum absolute ``wz`` sampled by the ``yaw`` and ``x_yaw`` roles [rad/s]."""
