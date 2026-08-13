# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the RoK4 ADAPT actuator model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import torch


class _IdealPDActuatorStub:
    def reset(self, env_ids) -> None:
        pass


class _DelayedPDActuatorStub(_IdealPDActuatorStub):
    pass


class _DelayedPDActuatorCfgStub:
    pass


class _DelayBufferStub:
    def __init__(self, history_length: int, batch_size: int, device: str):
        self.history_length = history_length
        self.batch_size = batch_size
        self.device = device
        self._time_lags = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._history: list[torch.Tensor] = []

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids=None) -> None:
        if batch_ids is None:
            batch_ids = slice(None)
        self._time_lags[batch_ids] = time_lag

    def reset(self, batch_ids=None) -> None:
        self._history.clear()

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        self._history.append(data.clone())
        self._history = self._history[-(self.history_length + 1) :]
        delayed = torch.empty_like(data)
        for batch_id in range(self.batch_size):
            available_lag = min(int(self._time_lags[batch_id]), len(self._history) - 1)
            delayed[batch_id] = self._history[-1 - available_lag][batch_id]
        return delayed


class _ArticulationActionsStub:
    def __init__(
        self,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor | None = None,
        joint_efforts: torch.Tensor | None = None,
    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_efforts = joint_efforts


_STUB_MODULES = {
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.actuators": ModuleType("isaaclab.actuators"),
    "isaaclab.utils": ModuleType("isaaclab.utils"),
    "isaaclab.utils.types": ModuleType("isaaclab.utils.types"),
}
_STUB_MODULES["isaaclab.actuators"].IdealPDActuator = _IdealPDActuatorStub
_STUB_MODULES["isaaclab.actuators"].IdealPDActuatorCfg = type("IdealPDActuatorCfg", (), {})
_STUB_MODULES["isaaclab.actuators"].DelayedPDActuator = _DelayedPDActuatorStub
_STUB_MODULES["isaaclab.actuators"].DelayedPDActuatorCfg = _DelayedPDActuatorCfgStub
_STUB_MODULES["isaaclab.utils"].DelayBuffer = _DelayBufferStub
_STUB_MODULES["isaaclab.utils"].configclass = lambda cls: cls
_STUB_MODULES["isaaclab.utils.types"].ArticulationActions = _ArticulationActionsStub

_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

_ADAPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/assets/robots/rok4_adapt.py"
)
_SPEC = importlib.util.spec_from_file_location("rok4_adapt_test_module", _ADAPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ADAPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ADAPT)
for _MODULE_NAME, _SAVED_MODULE in _SAVED_MODULES.items():
    if _SAVED_MODULE is None:
        del sys.modules[_MODULE_NAME]
    else:
        sys.modules[_MODULE_NAME] = _SAVED_MODULE


def test_adapt_pd_uses_two_step_delayed_position_target() -> None:
    """A two-step command lag must reach actuator-space PD before ADAPT torque mapping."""
    actuator = object.__new__(_ADAPT.RoK4AdaptActuator)
    actuator.positions_delay_buffer = _DelayBufferStub(2, 1, "cpu")
    actuator.positions_delay_buffer.set_time_lag(2)
    actuator.transmission = _ADAPT.RoK4AdaptTransmission(device="cpu")
    actuator._canonical_ids_in_model = torch.arange(13)
    actuator.stiffness = torch.ones((1, 13))
    actuator.damping = torch.zeros((1, 13))
    actuator.actuator_torque_limit = torch.full((1, 13), 100.0)
    actuator.computed_actuator_effort = torch.zeros((1, 13))
    actuator.applied_actuator_effort = torch.zeros((1, 13))
    actuator.computed_effort = torch.zeros((1, 13))
    actuator.applied_effort = torch.zeros((1, 13))

    joint_pos = torch.zeros((1, 13))
    joint_vel = torch.zeros((1, 13))
    applied_direct_drive_torques = []
    for target in (0.0, 1.0, 2.0, 3.0):
        joint_target = torch.zeros((1, 13))
        joint_target[:, 0] = target
        control_action = _ArticulationActionsStub(joint_positions=joint_target)
        result = actuator.compute(control_action, joint_pos, joint_vel)
        applied_direct_drive_torques.append(result.joint_efforts[0, 0].item())

    assert applied_direct_drive_torques == [0.0, 0.0, 0.0, 1.0]


def test_adapt_actuator_uses_delayed_pd_inheritance() -> None:
    """RoK4 actuator and config must expose Isaac Lab's standard delayed-PD API."""
    assert issubclass(_ADAPT.RoK4AdaptActuator, _DelayedPDActuatorStub)
    assert issubclass(_ADAPT.RoK4AdaptActuatorCfg, _DelayedPDActuatorCfgStub)
