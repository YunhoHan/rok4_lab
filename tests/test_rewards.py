# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RoK4-local reward functions."""

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


class _ManagerTermBaseStub:
    def __init__(self, cfg, env):
        pass


class _SceneStub(dict):
    def __init__(self, entities: dict, sensors: dict, env_origins: torch.Tensor | None = None):
        super().__init__(entities)
        self.sensors = sensors
        self.env_origins = env_origins


def _identity_quat_apply(_quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return vectors


_PACKAGE_NAME = "rok4_rewards_test_package"
_STUB_MODULES = {
    _PACKAGE_NAME: ModuleType(_PACKAGE_NAME),
    f"{_PACKAGE_NAME}.observations": ModuleType(f"{_PACKAGE_NAME}.observations"),
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.managers": ModuleType("isaaclab.managers"),
    "isaaclab.sensors": ModuleType("isaaclab.sensors"),
    "isaaclab.utils": ModuleType("isaaclab.utils"),
    "isaaclab.utils.math": ModuleType("isaaclab.utils.math"),
    "rok4_tasks": ModuleType("rok4_tasks"),
    "rok4_tasks.assets": ModuleType("rok4_tasks.assets"),
    "rok4_tasks.assets.robots": ModuleType("rok4_tasks.assets.robots"),
    "rok4_tasks.assets.robots.rok4": ModuleType("rok4_tasks.assets.robots.rok4"),
}
_STUB_MODULES[_PACKAGE_NAME].__path__ = []
_STUB_MODULES[f"{_PACKAGE_NAME}.observations"]._adapt_actuator = lambda asset, name: None
_STUB_MODULES["isaaclab.managers"].ManagerTermBase = _ManagerTermBaseStub
_STUB_MODULES["isaaclab.managers"].RewardTermCfg = object
_STUB_MODULES["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfgStub
_STUB_MODULES["isaaclab.sensors"].ContactSensor = object
_STUB_MODULES["isaaclab.utils.math"].quat_apply = _identity_quat_apply
_STUB_MODULES["isaaclab.utils.math"].quat_apply_inverse = _identity_quat_apply
_STUB_MODULES["isaaclab.utils.math"].yaw_quat = lambda quat: quat
_STUB_MODULES["rok4_tasks.assets.robots.rok4"].ROK4_JOINT_ORDER = tuple(f"joint_{index}" for index in range(13))

_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

_REWARDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/rewards.py"
)
_SPEC = importlib.util.spec_from_file_location(f"{_PACKAGE_NAME}.rewards", _REWARDS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_REWARDS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWARDS)
for _MODULE_NAME, _SAVED_MODULE in _SAVED_MODULES.items():
    if _SAVED_MODULE is None:
        del sys.modules[_MODULE_NAME]
    else:
        sys.modules[_MODULE_NAME] = _SAVED_MODULE


def test_lateral_separation_penalizes_narrow_and_crossed_feet_in_all_command_states() -> None:
    """Signed width must distinguish normal, narrow, and crossed configurations without command gating."""
    body_pos_w = torch.tensor(
        [
            [[0.0, 0.10, 0.0], [0.0, -0.10, 0.0]],
            [[0.0, 0.02, 0.0], [0.0, -0.02, 0.0]],
            [[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]],
            [[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]],
        ]
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=body_pos_w,
            root_quat_w=torch.zeros((4, 4)),
        )
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = _SceneEntityCfgStub("robot")

    penalty = _REWARDS.feet_lateral_separation_l2(
        env,
        minimum_width=0.16,
        asset_cfg=asset_cfg,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0144, 0.0676, 0.0676]))


def test_feet_air_time_rewards_single_touchdown_once_and_masks_small_commands() -> None:
    """Completed air time must be paid only for one-foot touchdown under a moving command."""

    class _ContactSensorStub:
        def __init__(self):
            self.data = SimpleNamespace(
                last_air_time=torch.tensor(
                    [
                        [0.80, 0.20],
                        [0.10, 0.40],
                        [0.60, 0.70],
                        [0.50, 0.20],
                        [0.50, 0.20],
                    ]
                )
            )

        def compute_first_contact(self, dt: float) -> torch.Tensor:
            assert dt == 0.01
            return torch.tensor(
                [
                    [True, False],
                    [False, True],
                    [True, True],
                    [True, False],
                    [False, False],
                ]
            )

    commands = torch.tensor(
        [
            [0.20, 0.00, 0.00],
            [0.00, 0.20, 0.00],
            [0.20, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.20, 0.00, 0.00],
        ]
    )
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(entities={}, sensors={"contact_forces": _ContactSensorStub()}),
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
    )

    reward = _REWARDS.feet_air_time_touchdown_biped(
        env,
        command_name="base_velocity",
        threshold=0.65,
        sensor_cfg=_SceneEntityCfgStub("contact_forces"),
    )

    torch.testing.assert_close(reward, torch.tensor([0.65, 0.40, 0.0, 0.0, 0.0]))


def test_base_height_uses_environment_relative_world_height() -> None:
    """Base-height error must be independent of each environment's world offset."""
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [4.0, 2.0, 2.0]])
    root_pos_w = env_origins.clone()
    root_pos_w[:, 2] += torch.tensor([0.907, 0.957, 0.807])
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=root_pos_w))
    env = SimpleNamespace(
        scene=_SceneStub(entities={"robot": asset}, sensors={}, env_origins=env_origins),
    )

    penalty = _REWARDS.base_height_relative_l2(
        env,
        target_height=0.907,
        asset_cfg=_SceneEntityCfgStub("robot"),
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0025, 0.01]))


def test_feet_swing_clearance_rewards_valid_moving_swing_feet() -> None:
    """Clearance reward must gate height shaping by speed, swing state, and standing role."""
    env_origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
        ]
    )
    relative_heights = torch.tensor(
        [
            [0.10, 0.004],
            [0.15, 0.004],
            [0.10, 0.10],
            [0.10, 0.004],
            [0.10, 0.004],
        ]
    )
    body_pos_w = torch.zeros((5, 2, 3))
    body_pos_w[..., 2] = relative_heights + env_origins[:, 2].unsqueeze(-1)
    body_lin_vel_w = torch.zeros((5, 2, 3))
    body_lin_vel_w[:, 0, 0] = 0.20
    body_lin_vel_w[2, 1, 0] = 0.40
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
        )
    )
    contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_air_time=torch.tensor(
                [
                    [0.30, 0.00],
                    [0.30, 0.00],
                    [0.30, 0.40],
                    [0.70, 0.00],
                    [0.30, 0.00],
                ]
            )
        )
    )
    command_term = SimpleNamespace(is_standing_env=torch.tensor([False, False, False, False, True]))
    env = SimpleNamespace(
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": contact_sensor},
            env_origins=env_origins,
        ),
        command_manager=SimpleNamespace(get_term=lambda _name: command_term),
    )

    reward = _REWARDS.feet_swing_clearance_exp(
        env,
        command_name="base_velocity",
        target_height=0.10,
        std=0.05,
        velocity_scale=0.20,
        asset_cfg=_SceneEntityCfgStub("robot"),
        sensor_cfg=_SceneEntityCfgStub("contact_forces"),
    )

    tanh_one = torch.tanh(torch.tensor(1.0))
    expected = torch.tensor(
        [
            tanh_one,
            tanh_one * torch.exp(torch.tensor(-1.0)),
            0.5 * (tanh_one + torch.tanh(torch.tensor(2.0))),
            tanh_one,
            0.0,
        ]
    )
    torch.testing.assert_close(reward, expected)


def test_feet_swing_roll_penalizes_only_airborne_feet(monkeypatch) -> None:
    """Swing-foot roll must be penalized without constraining feet that are in contact."""

    def _foot_up_from_quat(quat: torch.Tensor, _vectors: torch.Tensor) -> torch.Tensor:
        return quat[..., :3]

    def _identity_checked(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        assert quat.shape[:-1] == vectors.shape[:-1]
        return vectors

    monkeypatch.setattr(_REWARDS, "quat_apply", _foot_up_from_quat)
    monkeypatch.setattr(_REWARDS, "quat_apply_inverse", _identity_checked)

    body_quat_w = torch.tensor(
        [
            [[0.0, 0.5, 0.866, 0.0], [0.0, 0.8, 0.6, 0.0]],
            [[0.0, 0.5, 0.866, 0.0], [0.0, 0.7, 0.714, 0.0]],
            [[0.0, 0.5, 0.866, 0.0], [0.0, 0.3, 0.954, 0.0]],
        ]
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_quat_w=body_quat_w,
            root_quat_w=torch.zeros((3, 4)),
        )
    )
    contact_sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_contact_time=torch.tensor(
                [
                    [0.0, 0.1],
                    [0.1, 0.1],
                    [0.0, 0.0],
                ]
            )
        )
    )
    env = SimpleNamespace(
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": contact_sensor},
        )
    )
    asset_cfg = _SceneEntityCfgStub("robot")
    sensor_cfg = _SceneEntityCfgStub("contact_forces")

    penalty = _REWARDS.feet_swing_roll_l2(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.25, 0.0, 0.17]))
