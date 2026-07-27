# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RoK4 episode command roles."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import torch


class _UniformVelocityCommandStub:
    @property
    def device(self) -> str:
        return self._env.device


class _UniformVelocityCommandCfgStub:
    pass


def _configclass_stub(cls):
    return cls


_STUB_MODULES = {
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.envs": ModuleType("isaaclab.envs"),
    "isaaclab.envs.mdp": ModuleType("isaaclab.envs.mdp"),
    "isaaclab.envs.mdp.commands": ModuleType("isaaclab.envs.mdp.commands"),
    "isaaclab.utils": ModuleType("isaaclab.utils"),
}
_STUB_MODULES["isaaclab.envs.mdp.commands"].UniformVelocityCommand = _UniformVelocityCommandStub
_STUB_MODULES["isaaclab.envs.mdp.commands"].UniformVelocityCommandCfg = _UniformVelocityCommandCfgStub
_STUB_MODULES["isaaclab.utils"].configclass = _configclass_stub
_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

_COMMANDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/commands.py"
)
_SPEC = importlib.util.spec_from_file_location("rok4_commands_under_test", _COMMANDS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_COMMANDS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMANDS)
for _MODULE_NAME, _SAVED_MODULE in _SAVED_MODULES.items():
    if _SAVED_MODULE is None:
        del sys.modules[_MODULE_NAME]
    else:
        sys.modules[_MODULE_NAME] = _SAVED_MODULE


def _command_stub(num_envs: int = 128):
    command = object.__new__(_COMMANDS.RoK4PeriodicFreezeVelocityCommand)
    command._env = SimpleNamespace(device="cpu")
    command.cfg = SimpleNamespace(
        mixed_env_ratio=0.50,
        standing_env_ratio=0.05,
        walking_env_ratio=0.05,
        x_env_ratio=0.10,
        y_env_ratio=0.10,
        yaw_env_ratio=0.10,
        x_yaw_env_ratio=0.10,
        dedicated_x_min_abs_vel=0.15,
        dedicated_y_min_abs_vel=0.15,
        dedicated_yaw_min_abs_vel=0.15,
        ranges=SimpleNamespace(
            lin_vel_x=(-0.3, 0.85),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.6, 0.6),
        ),
    )
    command.vel_command_b = torch.full((num_envs, 3), torch.nan)
    command.is_standing_env = torch.zeros(num_envs, dtype=torch.bool)
    command._environment_role = torch.zeros(num_envs, dtype=torch.long)
    command._is_periodic_freeze_env = torch.zeros(num_envs, dtype=torch.bool)
    command._periodic_freeze_active = torch.zeros(num_envs, dtype=torch.bool)
    command._role_boundaries = torch.cumsum(torch.tensor(command._role_ratios(command.cfg)), dim=0)[:-1]
    return command


def test_episode_role_boundaries_and_freeze_eligibility(monkeypatch) -> None:
    """Every configured role must occupy its interval and use the intended freeze behavior."""
    command = _command_stub(num_envs=7)
    samples = torch.tensor([0.25, 0.525, 0.575, 0.65, 0.75, 0.85, 0.95])
    monkeypatch.setattr(torch, "rand", lambda count, device: samples.to(device))

    command._sample_environment_roles(torch.arange(7))

    torch.testing.assert_close(command._environment_role, torch.arange(7))
    torch.testing.assert_close(
        command._is_periodic_freeze_env,
        torch.tensor([True, False, False, True, True, True, True]),
    )


def test_dedicated_roles_activate_only_their_command_axes() -> None:
    """Dedicated x, y, yaw, and x-yaw roles must zero every unrelated command axis."""
    command = _command_stub()
    env_ids = torch.arange(128)

    command._resample_x_command(env_ids)
    assert torch.all(command.vel_command_b[:, 1:] == 0.0)
    assert torch.all((torch.abs(command.vel_command_b[:, 0]) >= 0.15))
    assert torch.all((command.vel_command_b[:, 0] >= -0.3) & (command.vel_command_b[:, 0] <= 0.85))

    command._resample_y_command(env_ids)
    assert torch.all(command.vel_command_b[:, [0, 2]] == 0.0)
    assert torch.all((torch.abs(command.vel_command_b[:, 1]) >= 0.15))
    assert torch.all((command.vel_command_b[:, 1] >= -0.3) & (command.vel_command_b[:, 1] <= 0.3))

    command._resample_yaw_command(env_ids)
    assert torch.all(command.vel_command_b[:, :2] == 0.0)
    assert torch.all((torch.abs(command.vel_command_b[:, 2]) >= 0.15))
    assert torch.all((command.vel_command_b[:, 2] >= -0.6) & (command.vel_command_b[:, 2] <= 0.6))

    command._resample_x_yaw_command(env_ids)
    assert torch.all(command.vel_command_b[:, 1] == 0.0)
    assert torch.all(torch.abs(command.vel_command_b[:, 0]) >= 0.15)
    assert torch.all(torch.abs(command.vel_command_b[:, 2]) >= 0.15)


def test_balanced_sampler_generates_both_signs() -> None:
    """A sufficiently large dedicated-role sample must cover positive and negative commands."""
    command = _command_stub()
    torch.manual_seed(7)

    values = command._sample_balanced_signed_values(4096, (-0.3, 0.85), 0.15)

    positive_ratio = torch.mean((values > 0.0).float()).item()
    assert 0.47 < positive_ratio < 0.53
    assert torch.all((values <= -0.15) | (values >= 0.15))
    assert torch.all((values >= -0.3) & (values <= 0.85))


def test_standing_mask_combines_fixed_and_periodic_standing() -> None:
    """The standing reward mask must include fixed-standing and active freeze environments."""
    command = _command_stub(num_envs=4)
    command.vel_command_b[:] = 1.0
    command._environment_role[:] = torch.tensor(
        [command._ROLE_STANDING, command._ROLE_WALKING, command._ROLE_X, command._ROLE_Y]
    )
    command._periodic_freeze_active[2] = True

    command._apply_standing_mask(torch.arange(4))

    torch.testing.assert_close(command.is_standing_env, torch.tensor([True, False, True, False]))
    assert torch.all(command.vel_command_b[[0, 2]] == 0.0)
    assert torch.all(command.vel_command_b[[1, 3]] == 1.0)
