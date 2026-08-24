# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the RoK4 manual push coordinate conversion."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _quat_apply_z(quat_w: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the yaw-only WXYZ quaternions used by this test."""
    yaw = 2.0 * torch.atan2(quat_w[:, 3], quat_w[:, 0])
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    result = vectors.clone()
    result[:, 0] = cosine * vectors[:, 0] - sine * vectors[:, 1]
    result[:, 1] = sine * vectors[:, 0] + cosine * vectors[:, 1]
    return result


_STUB_MODULES = {
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.envs": ModuleType("isaaclab.envs"),
    "isaaclab.envs.ui": ModuleType("isaaclab.envs.ui"),
    "isaaclab.utils": ModuleType("isaaclab.utils"),
    "isaaclab.utils.math": ModuleType("isaaclab.utils.math"),
}


class _WindowBaseStub:
    def __del__(self) -> None:
        pass


_STUB_MODULES["isaaclab.envs.ui"].ManagerBasedRLEnvWindow = _WindowBaseStub
_STUB_MODULES["isaaclab.utils.math"].quat_apply = _quat_apply_z
_STUB_MODULES["isaaclab.utils.math"].quat_apply_inverse = lambda quat, vectors: vectors
_STUB_MODULES["isaaclab.utils.math"].yaw_quat = lambda quat: quat

_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

_WINDOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/config/rok4/push_test_window.py"
)
_SPEC = importlib.util.spec_from_file_location("rok4_push_test_window_under_test", _WINDOW_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WINDOW_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WINDOW_MODULE)
for _MODULE_NAME, _SAVED_MODULE in _SAVED_MODULES.items():
    if _SAVED_MODULE is None:
        del sys.modules[_MODULE_NAME]
    else:
        sys.modules[_MODULE_NAME] = _SAVED_MODULE


class _RobotStub:
    def __init__(self) -> None:
        half_yaw = math.pi / 4.0
        self.data = SimpleNamespace(
            root_vel_w=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            root_quat_w=torch.tensor([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]]),
        )
        self.written_velocity = None
        self.written_env_ids = None

    def write_root_velocity_to_sim(self, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.written_velocity = velocity
        self.written_env_ids = env_ids


def test_manual_push_rotates_base_forward_by_current_yaw() -> None:
    """A base-forward push at 90-degree yaw must become world-left without changing other components."""
    robot = _RobotStub()
    window = _WINDOW_MODULE.RoK4PushTestWindow.__new__(_WINDOW_MODULE.RoK4PushTestWindow)
    window.env = SimpleNamespace(device="cpu", scene={"robot": robot})
    window._pending_push_delta_b = (0.5, 0.0)
    window._push_status_label = None
    window._selected_env_index = lambda: 0

    assert window.apply_pending_push()

    torch.testing.assert_close(
        robot.written_velocity,
        torch.tensor([[1.0, 2.5, 3.0, 4.0, 5.0, 6.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(robot.written_env_ids, torch.tensor([0]))
    assert window._pending_push_delta_b is None
