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

import pytest
import torch


class _SceneEntityCfgStub:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.body_names = kwargs.get("body_names")
        self.body_ids = kwargs.get("body_ids", slice(None))


class _ManagerTermBaseStub:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


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


def test_feet_air_time_rewards_signed_single_touchdown_and_masks_small_commands() -> None:
    """Completed air time must be shaped once for a moving one-foot touchdown."""

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
                    [True, False],
                    [False, False],
                ]
            )

    commands = torch.tensor(
        [
            [0.20, 0.00, 0.00],
            [0.00, 0.20, 0.00],
            [0.20, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.075, 0.00, 0.00],
            [0.20, 0.00, 0.00],
        ]
    )
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(entities={}, sensors={"contact_forces": _ContactSensorStub()}),
        command_manager=SimpleNamespace(get_command=lambda _name: commands),
        num_envs=6,
        device="cpu",
        extras={"log": {}},
    )
    sensor_cfg = _SceneEntityCfgStub("contact_forces", body_ids=[0, 1])

    reward = _REWARDS.feet_air_time_touchdown_biped(
        env,
        command_name="base_velocity",
        target_air_time=0.50,
        sensor_cfg=sensor_cfg,
        command_threshold=0.05,
    )

    torch.testing.assert_close(reward, torch.tensor([0.30, -0.10, 0.0, 0.0, 0.0, 0.0]))

    term = _REWARDS.FeetAirTimeTouchdownBiped(
        SimpleNamespace(
            params={
                "command_name": "base_velocity",
                "target_air_time": 0.50,
                "sensor_cfg": sensor_cfg,
                "command_threshold": 0.05,
            }
        ),
        env,
    )
    class_reward = term(
        env,
        command_name="base_velocity",
        target_air_time=0.50,
        sensor_cfg=sensor_cfg,
        command_threshold=0.05,
    )
    torch.testing.assert_close(class_reward, reward)

    term.reset(torch.arange(6))
    torch.testing.assert_close(
        env.extras["log"]["Metrics/feet_touchdown/mean_air_time"],
        torch.tensor((0.80 + 0.40 + 0.50) / 3.0),
    )
    torch.testing.assert_close(term._air_time_sum, torch.zeros(6))
    torch.testing.assert_close(term._touchdown_count, torch.zeros(6))


def test_feet_air_time_rejects_invalid_timing_parameters() -> None:
    """Signed air-time shaping must reject invalid target and command thresholds."""
    with pytest.raises(ValueError, match="target_air_time must be non-negative"):
        _REWARDS.feet_air_time_touchdown_biped(
            SimpleNamespace(),
            command_name="base_velocity",
            target_air_time=-0.01,
            sensor_cfg=_SceneEntityCfgStub("contact_forces"),
        )

    with pytest.raises(ValueError, match="command_threshold must be non-negative"):
        _REWARDS.feet_air_time_touchdown_biped(
            SimpleNamespace(),
            command_name="base_velocity",
            target_air_time=0.50,
            sensor_cfg=_SceneEntityCfgStub("contact_forces"),
            command_threshold=-0.01,
        )


def test_action_rate_uses_unclipped_raw_actions() -> None:
    """First-order action rate must preserve policy outputs beyond the actuator clip range."""
    action = torch.zeros((1, 13))
    prev_action = torch.zeros_like(action)
    action[0, 0] = 1.5
    prev_action[0, 0] = 0.5
    action[0, 2] = 1.5
    prev_action[0, 2] = 0.5
    env = SimpleNamespace(
        action_manager=SimpleNamespace(action=action, prev_action=prev_action),
    )

    penalty = _REWARDS.action_rate_l2(env)

    torch.testing.assert_close(penalty, torch.tensor([1.5]))


def test_second_action_rate_uses_unclipped_raw_actions() -> None:
    """Second-order action rate must use raw action history without clipping."""
    action_manager = SimpleNamespace(
        action=torch.zeros((1, 13)),
        prev_action=torch.zeros((1, 13)),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", action_manager=action_manager)
    term = _REWARDS.second_action_rate_l2(SimpleNamespace(params={}), env)

    action_manager.action[0, 0] = 0.1
    torch.testing.assert_close(term(env), torch.zeros(1))

    action_manager.prev_action[0, 0] = 0.1
    action_manager.action[0, 0] = 0.2
    torch.testing.assert_close(term(env), torch.zeros(1))

    action_manager.prev_action[0, 0] = 0.2
    action_manager.action[0, 0] = 1.5
    torch.testing.assert_close(term(env), torch.tensor([1.44]))


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


def test_feet_touchdown_acc_penalizes_only_excessive_first_contact_acceleration() -> None:
    """Foot acceleration must be thresholded and evaluated only on first-contact events."""

    class _ContactSensorStub:
        def compute_first_contact(self, dt: float) -> torch.Tensor:
            assert dt == 0.01
            return torch.tensor(
                [
                    [True, False],
                    [False, True],
                    [True, False],
                    [False, False],
                    [True, True],
                ]
            )

    body_lin_acc_w = torch.tensor(
        [
            [[30.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            [[100.0, 0.0, 0.0], [30.0, 40.0, 0.0]],
            [[0.0, 0.0, 80.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
            [[0.0, 0.0, 60.0], [0.0, 0.0, 100.0]],
        ]
    )
    asset = SimpleNamespace(data=SimpleNamespace(body_lin_acc_w=body_lin_acc_w))
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": _ContactSensorStub()},
        ),
    )
    body_ids = [0, 1]

    penalty = _REWARDS.feet_touchdown_acc(
        env,
        sensor_cfg=_SceneEntityCfgStub("contact_forces", body_ids=body_ids),
        asset_cfg=_SceneEntityCfgStub("robot", body_ids=body_ids),
        threshold=50.0,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.0, 0.0, 30.0, 0.0, 60.0]))


def test_feet_touchdown_acc_rejects_negative_threshold() -> None:
    """Touchdown acceleration threshold must remain physically meaningful."""
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        _REWARDS.feet_touchdown_acc(
            SimpleNamespace(),
            sensor_cfg=_SceneEntityCfgStub("contact_forces"),
            asset_cfg=_SceneEntityCfgStub("robot"),
            threshold=-1.0,
        )


def test_feet_touchdown_velocity_uses_previous_airborne_speed_once() -> None:
    """Touchdown shaping must use the preceding airborne sample, not post-impact velocity."""

    class _ContactSensorStub:
        def __init__(self):
            self.data = SimpleNamespace(
                current_contact_time=torch.tensor(
                    [[0.0, 0.0], [0.0, 0.0], [0.01, 0.0]]
                )
            )
            self.first_contact = torch.tensor(
                [[False, False], [False, False], [True, False]]
            )

        def compute_first_contact(self, dt: float) -> torch.Tensor:
            assert dt == 0.01
            return self.first_contact

    body_lin_vel_w = torch.zeros((3, 2, 3))
    body_lin_vel_w[..., 2] = torch.tensor(
        [[-0.6, -0.1], [-0.1, -0.5], [-1.0, 0.0]]
    )
    asset = SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=body_lin_vel_w))
    sensor = _ContactSensorStub()
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": sensor},
        ),
        num_envs=3,
        device="cpu",
        extras={"log": {}},
    )
    sensor_cfg = _SceneEntityCfgStub("contact_forces", body_ids=[0, 1])
    asset_cfg = _SceneEntityCfgStub("robot", body_ids=[0, 1])
    term = _REWARDS.FeetTouchdownVelocityL2(
        SimpleNamespace(
            params={
                "sensor_cfg": sensor_cfg,
                "asset_cfg": asset_cfg,
                "safe_landing_velocity": 0.0,
            }
        ),
        env,
    )

    initial_penalty = term(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        safe_landing_velocity=0.0,
    )
    torch.testing.assert_close(initial_penalty, torch.zeros(3))

    body_lin_vel_w[..., 2] = 0.0
    sensor.data.current_contact_time = torch.tensor(
        [[0.01, 0.0], [0.0, 0.01], [0.01, 0.0]]
    )
    sensor.first_contact = torch.tensor(
        [[True, False], [False, True], [True, False]]
    )
    touchdown_penalty = term(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        safe_landing_velocity=0.0,
    )

    torch.testing.assert_close(touchdown_penalty, torch.tensor([0.36, 0.25, 0.0]))
    term.reset(torch.arange(3))
    torch.testing.assert_close(
        env.extras["log"]["Metrics/feet_touchdown/mean_pre_touchdown_vertical_speed"],
        torch.tensor(0.55),
    )
    torch.testing.assert_close(term._previous_foot_vel_z, torch.zeros((3, 2)))
    assert not torch.any(term._has_previous_sample)


def test_feet_touchdown_velocity_rejects_negative_safe_speed() -> None:
    """The permitted pre-touchdown speed must be non-negative."""
    with pytest.raises(ValueError, match="safe_landing_velocity must be non-negative"):
        _REWARDS.FeetTouchdownVelocityL2(
            SimpleNamespace(
                params={
                    "sensor_cfg": _SceneEntityCfgStub("contact_forces", body_ids=[0, 1]),
                    "asset_cfg": _SceneEntityCfgStub("robot", body_ids=[0, 1]),
                    "safe_landing_velocity": -0.1,
                }
            ),
            SimpleNamespace(num_envs=1, device="cpu"),
        )


def test_feet_contact_velocity_penalizes_fast_approach_and_touchdown() -> None:
    """Soft-landing velocity shaping must act immediately before and at contact."""

    class _ContactSensorStub:
        def __init__(self):
            self.data = SimpleNamespace(
                current_contact_time=torch.tensor(
                    [
                        [0.0, 0.0],
                        [0.2, 0.01],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                )
            )

        def compute_first_contact(self, dt: float) -> torch.Tensor:
            assert dt == 0.01
            return torch.tensor(
                [
                    [False, False],
                    [False, True],
                    [True, False],
                    [False, False],
                ]
            )

    env_origins = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]]
    )
    relative_heights = torch.tensor(
        [[0.02, 0.04], [0.02, 0.02], [0.02, 0.04], [0.031, 0.02]]
    )
    body_pos_w = torch.zeros((4, 2, 3))
    body_pos_w[..., 2] = relative_heights + env_origins[:, 2].unsqueeze(-1)
    body_lin_vel_w = torch.zeros((4, 2, 3))
    body_lin_vel_w[..., 2] = torch.tensor(
        [[-0.8, -1.0], [-1.0, -0.4], [0.2, -0.8], [-0.8, -0.5]]
    )
    asset = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=body_pos_w, body_lin_vel_w=body_lin_vel_w)
    )
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": _ContactSensorStub()},
            env_origins=env_origins,
        ),
    )

    penalty = _REWARDS.feet_contact_velocity_l2(
        env,
        sensor_cfg=_SceneEntityCfgStub("contact_forces", body_ids=[0, 1]),
        asset_cfg=_SceneEntityCfgStub("robot", body_ids=[0, 1]),
        landing_height=0.03,
        approach_velocity_threshold=-0.6,
        impact_velocity_threshold=0.0,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.64, 0.16, 0.0, 0.0]))


def test_feet_contact_force_uses_peak_history_and_hybrid_weights() -> None:
    """Contact-force shaping must retain impact peaks and distinguish landing from stance."""

    class _ContactSensorStub:
        def __init__(self):
            force_history = torch.zeros((4, 5, 2, 1, 3))
            force_history[0, 3, :, 0, 2] = torch.tensor([800.0, 1200.0])
            force_history[1, 2, :, 0, 2] = torch.tensor([1100.0, 2000.0])
            force_history[2, 4, :, 0, 2] = torch.tensor([1500.0, 1250.0])
            force_history[3, 1, :, 0, 2] = torch.tensor([3000.0, 3000.0])
            self.data = SimpleNamespace(
                force_matrix_w_history=force_history,
                current_contact_time=torch.tensor(
                    [[0.2, 0.01], [0.2, 0.2], [0.01, 0.01], [0.0, 0.0]]
                ),
            )

        def compute_first_contact(self, dt: float) -> torch.Tensor:
            assert dt == 0.01
            return torch.tensor(
                [[False, True], [False, False], [True, True], [False, False]]
            )

    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(entities={}, sensors={"contact_forces": _ContactSensorStub()}),
    )
    penalty = _REWARDS.feet_contact_force_l2(
        env,
        sensor_cfg=_SceneEntityCfgStub("contact_forces", body_ids=[0, 1]),
        max_contact_force=1000.0,
        stance_weight=0.15,
        landing_weight=1.0,
    )

    torch.testing.assert_close(
        penalty,
        torch.tensor([0.04, 0.1515, 0.3125, 0.0]),
    )


def test_feet_contact_force_class_uses_body_weight_and_landing_window() -> None:
    """The stateful force term must penalize the 50 ms landing peak once using 1.5 BW."""

    class _ContactSensorStub:
        def __init__(self):
            self.data = SimpleNamespace()
            self._first_contact = torch.zeros((2, 2), dtype=torch.bool)

        def set_step(
            self,
            force_z: torch.Tensor,
            contact_time: float,
            first_contact: bool = False,
        ) -> None:
            force_history = torch.zeros((2, 5, 2, 1, 3))
            force_history[:, 3, 0, 0, 2] = force_z
            self.data.force_matrix_w_history = force_history
            self.data.current_contact_time = torch.tensor(
                [[contact_time, 0.0], [contact_time, 0.0]]
            )
            self._first_contact.zero_()
            self._first_contact[:, 0] = first_contact

        def compute_first_contact(self, dt: float) -> torch.Tensor:
            assert dt == 0.01
            return self._first_contact

    class _RootPhysXViewStub:
        @staticmethod
        def get_masses() -> torch.Tensor:
            # Total masses are 50 kg and 100 kg, respectively.
            return torch.tensor([[20.0, 30.0], [40.0, 60.0]])

    sensor_cfg = _SceneEntityCfgStub("contact_forces", body_ids=[0, 1])
    asset_cfg = _SceneEntityCfgStub("robot")
    contact_sensor = _ContactSensorStub()
    asset = SimpleNamespace(root_physx_view=_RootPhysXViewStub())
    env = SimpleNamespace(
        step_dt=0.01,
        scene=_SceneStub(entities={"robot": asset}, sensors={"contact_forces": contact_sensor}),
        sim=SimpleNamespace(cfg=SimpleNamespace(gravity=(0.0, 0.0, -10.0))),
        num_envs=2,
        device="cpu",
        extras={"log": {}},
    )
    term = _REWARDS.FeetContactForceL2(
        SimpleNamespace(
            params={
                "sensor_cfg": sensor_cfg,
                "asset_cfg": asset_cfg,
                "force_limit_multiplier": 1.5,
                "landing_window_s": 0.05,
            }
        ),
        env,
    )

    contact_sensor.set_step(torch.tensor([600.0, 1800.0]), 0.01, first_contact=True)
    penalty_at_touchdown = term(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        force_limit_multiplier=1.5,
        landing_window_s=0.05,
    )
    torch.testing.assert_close(penalty_at_touchdown, torch.zeros(2))

    contact_sensor.set_step(torch.tensor([900.0, 1600.0]), 0.03)
    penalty_during_window = term(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        force_limit_multiplier=1.5,
        landing_window_s=0.05,
    )
    torch.testing.assert_close(penalty_during_window, torch.zeros(2))

    contact_sensor.set_step(torch.tensor([1200.0, 2000.0]), 0.05)
    penalty_at_window_end = term(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        force_limit_multiplier=1.5,
        landing_window_s=0.05,
    )
    torch.testing.assert_close(
        penalty_at_window_end,
        torch.tensor([0.36, 1.0 / 9.0]),
    )
    torch.testing.assert_close(term._robot_weight, torch.tensor([500.0, 1000.0]))

    term.reset(torch.arange(2))
    torch.testing.assert_close(
        env.extras["log"]["Metrics/feet_touchdown/mean_peak_normal_force"],
        torch.tensor(1600.0),
    )
    torch.testing.assert_close(term._touchdown_force_sum, torch.zeros(2))
    torch.testing.assert_close(term._touchdown_count, torch.zeros(2))
    assert not torch.any(term._landing_active)
    torch.testing.assert_close(term._landing_peak_force, torch.zeros((2, 2)))

    metric_term = _REWARDS.FeetContactForceL2(
        SimpleNamespace(
            params={
                "sensor_cfg": sensor_cfg,
                "asset_cfg": asset_cfg,
                "force_limit_multiplier": 1.5,
                "landing_window_s": 0.05,
                "metric_only": True,
            }
        ),
        env,
    )
    for force_z, contact_time, first_contact in (
        (torch.tensor([600.0, 1800.0]), 0.01, True),
        (torch.tensor([900.0, 1600.0]), 0.03, False),
        (torch.tensor([1200.0, 2000.0]), 0.05, False),
    ):
        contact_sensor.set_step(force_z, contact_time, first_contact=first_contact)
        metric_value = metric_term(
            env,
            sensor_cfg=sensor_cfg,
            asset_cfg=asset_cfg,
            force_limit_multiplier=1.5,
            landing_window_s=0.05,
            metric_only=True,
        )
        torch.testing.assert_close(metric_value, torch.zeros(2))

    assert metric_term._robot_weight is None
    metric_term.reset(torch.arange(2))
    torch.testing.assert_close(
        env.extras["log"]["Metrics/feet_touchdown/mean_peak_normal_force"],
        torch.tensor(1600.0),
    )


@pytest.mark.parametrize(
    ("function_name", "params", "match"),
    [
        (
            "feet_contact_velocity_l2",
            {"landing_height": -0.01},
            "landing_height must be non-negative",
        ),
        (
            "feet_contact_force_l2",
            {"max_contact_force": 0.0},
            "max_contact_force must be positive",
        ),
    ],
)
def test_soft_landing_rewards_reject_invalid_parameters(
    function_name: str, params: dict, match: str
) -> None:
    """Soft-landing thresholds and force limits must remain physically meaningful."""
    function = getattr(_REWARDS, function_name)
    call_kwargs = {
        "sensor_cfg": _SceneEntityCfgStub("contact_forces"),
        **params,
    }
    if "velocity" in function_name:
        call_kwargs["asset_cfg"] = _SceneEntityCfgStub("robot")
    with pytest.raises(ValueError, match=match):
        function(SimpleNamespace(), **call_kwargs)


def test_feet_swing_clearance_rewards_valid_moving_swing_feet() -> None:
    """Clearance reward must gate height shaping by command-directed swing speed."""
    env_origins = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
            [5.0, 0.0, 5.0],
        ]
    )
    relative_heights = torch.tensor(
        [
            [0.075, 0.004],
            [0.115, 0.004],
            [0.075, 0.075],
            [0.075, 0.004],
            [0.075, 0.004],
            [0.075, 0.004],
        ]
    )
    body_pos_w = torch.zeros((6, 2, 3))
    body_pos_w[..., 2] = relative_heights + env_origins[:, 2].unsqueeze(-1)
    body_lin_vel_w = torch.zeros((6, 2, 3))
    body_lin_vel_w[:, 0, 0] = 0.20
    body_lin_vel_w[2, 1, 0] = 0.40
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=body_pos_w,
            body_lin_vel_w=body_lin_vel_w,
            root_quat_w=torch.zeros((6, 4)),
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
                    [0.30, 0.00],
                ]
            )
        )
    )
    command_term = SimpleNamespace(
        is_standing_env=torch.tensor([False, False, False, False, True, False])
    )
    commands = torch.tensor(
        [
            [0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.3],
        ]
    )
    env = SimpleNamespace(
        scene=_SceneStub(
            entities={"robot": asset},
            sensors={"contact_forces": contact_sensor},
            env_origins=env_origins,
        ),
        command_manager=SimpleNamespace(
            get_term=lambda _name: command_term,
            get_command=lambda _name: commands,
        ),
    )

    reward = _REWARDS.feet_swing_clearance_exp(
        env,
        command_name="base_velocity",
        target_height=0.075,
        std=0.04,
        velocity_scale=0.50,
        asset_cfg=_SceneEntityCfgStub("robot"),
        sensor_cfg=_SceneEntityCfgStub("contact_forces"),
    )

    tanh_point_four = torch.tanh(torch.tensor(0.4))
    expected = torch.tensor(
        [
            tanh_point_four,
            tanh_point_four * torch.exp(torch.tensor(-1.0)),
            0.5 * (tanh_point_four + torch.tanh(torch.tensor(0.8))),
            0.0,
            0.0,
            tanh_point_four,
        ]
    )
    torch.testing.assert_close(reward, expected)


@pytest.mark.parametrize(
    ("function_name", "tilt_axis"),
    (("feet_swing_roll_l2", 1), ("feet_swing_pitch_l2", 0)),
)
def test_feet_swing_tilt_penalizes_only_airborne_feet(
    monkeypatch, function_name: str, tilt_axis: int
) -> None:
    """Swing-foot tilt must be penalized without constraining feet that are in contact."""

    def _foot_up_from_quat(quat: torch.Tensor, _vectors: torch.Tensor) -> torch.Tensor:
        return quat[..., :3]

    def _identity_checked(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        assert quat.shape[:-1] == vectors.shape[:-1]
        return vectors

    monkeypatch.setattr(_REWARDS, "quat_apply", _foot_up_from_quat)
    monkeypatch.setattr(_REWARDS, "quat_apply_inverse", _identity_checked)

    tilt_values = torch.tensor(
        [
            [0.5, 0.8],
            [0.5, 0.7],
            [0.5, 0.3],
        ]
    )
    body_quat_w = torch.zeros((3, 2, 4))
    body_quat_w[..., tilt_axis] = tilt_values
    body_quat_w[..., 2] = torch.sqrt(1.0 - torch.square(tilt_values))
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

    penalty = getattr(_REWARDS, function_name)(
        env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )

    torch.testing.assert_close(penalty, torch.tensor([0.25, 0.0, 0.17]))
