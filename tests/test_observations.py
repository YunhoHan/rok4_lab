# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RoK4-local privileged observation functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import torch


class _SceneEntityCfgStub:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.body_ids = kwargs.get("body_ids", slice(None))


class _SceneStub(dict):
    def __init__(self, entities: dict, sensors: dict, env_origins: torch.Tensor):
        super().__init__(entities)
        self.sensors = sensors
        self.env_origins = env_origins


_ADAPT_MODULE = "rok4_tasks.assets.robots.rok4_adapt"
_STUB_MODULES = {
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.assets": ModuleType("isaaclab.assets"),
    "isaaclab.managers": ModuleType("isaaclab.managers"),
    "isaaclab.sensors": ModuleType("isaaclab.sensors"),
    "rok4_tasks": ModuleType("rok4_tasks"),
    "rok4_tasks.assets": ModuleType("rok4_tasks.assets"),
    "rok4_tasks.assets.robots": ModuleType("rok4_tasks.assets.robots"),
    _ADAPT_MODULE: ModuleType(_ADAPT_MODULE),
}
_STUB_MODULES["isaaclab.assets"].Articulation = object
_STUB_MODULES["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfgStub
_STUB_MODULES["isaaclab.sensors"].ContactSensor = object
_STUB_MODULES[_ADAPT_MODULE].RoK4AdaptActuator = type("RoK4AdaptActuator", (), {})

_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

_OBSERVATIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/observations.py"
)
_SPEC = importlib.util.spec_from_file_location("rok4_observations_under_test", _OBSERVATIONS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_OBSERVATIONS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OBSERVATIONS)
for _MODULE_NAME, _SAVED_MODULE in _SAVED_MODULES.items():
    if _SAVED_MODULE is None:
        del sys.modules[_MODULE_NAME]
    else:
        sys.modules[_MODULE_NAME] = _SAVED_MODULE


def test_current_privileged_observations_use_environment_relative_heights_and_lr_order() -> None:
    """Height, contact, and air-time terms must preserve the selected left-right body order."""
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    root_pos_w = torch.tensor([[0.0, 0.0, 0.907], [2.0, 1.0, 1.930]])
    body_pos_w = torch.tensor(
        [
            [[0.0, 0.1, 0.10], [0.0, -0.1, 0.04]],
            [[2.0, 1.1, 1.20], [2.0, 0.9, 1.08]],
        ]
    )
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos_w, body_pos_w=body_pos_w))
    contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_contact_time=torch.tensor([[0.0, 0.2], [0.1, 0.0]]),
            current_air_time=torch.tensor([[0.3, 0.0], [0.0, 0.4]]),
        )
    )
    env = SimpleNamespace(
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": contact_sensor},
            env_origins=env_origins,
        )
    )

    base_height = _OBSERVATIONS.base_height(env, _SceneEntityCfgStub("robot"))
    foot_height = _OBSERVATIONS.foot_height(env, _SceneEntityCfgStub("robot"))
    contact = _OBSERVATIONS.foot_contact_flag(env, _SceneEntityCfgStub("contact_forces"))
    air_time = _OBSERVATIONS.foot_current_air_time(env, _SceneEntityCfgStub("contact_forces"))

    torch.testing.assert_close(base_height, torch.tensor([[0.907], [0.930]]))
    torch.testing.assert_close(foot_height, torch.tensor([[0.10, 0.04], [0.20, 0.08]]))
    torch.testing.assert_close(contact, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    torch.testing.assert_close(air_time, torch.tensor([[0.3, 0.0], [0.0, 0.4]]))
