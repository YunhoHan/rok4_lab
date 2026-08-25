# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RoK4 reset events."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


class _SceneEntityCfgStub:
    def __init__(self, name: str):
        self.name = name
        self.joint_ids = slice(None)
        self.body_ids = slice(None)


class _ManagerTermBaseStub:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> str:
        return self._env.device


_MATH_UTILS_STUB = ModuleType("isaaclab.utils.math")
_STUB_MODULES = {
    "isaaclab": ModuleType("isaaclab"),
    "isaaclab.assets": ModuleType("isaaclab.assets"),
    "isaaclab.envs": ModuleType("isaaclab.envs"),
    "isaaclab.envs.mdp": ModuleType("isaaclab.envs.mdp"),
    "isaaclab.envs.mdp.events": ModuleType("isaaclab.envs.mdp.events"),
    "isaaclab.managers": ModuleType("isaaclab.managers"),
    "isaaclab.utils": ModuleType("isaaclab.utils"),
    "isaaclab.utils.math": _MATH_UTILS_STUB,
}
_STUB_MODULES["isaaclab.assets"].Articulation = object
_STUB_MODULES["isaaclab.envs.mdp.events"].randomize_rigid_body_material = object
_STUB_MODULES["isaaclab.managers"].EventTermCfg = object
_STUB_MODULES["isaaclab.managers"].ManagerTermBase = _ManagerTermBaseStub
_STUB_MODULES["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfgStub
_STUB_MODULES["isaaclab.utils"].math = _MATH_UTILS_STUB
_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULES}
sys.modules.update(_STUB_MODULES)

_EVENTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/events.py"
)
_SPEC = importlib.util.spec_from_file_location("rok4_events_under_test", _EVENTS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EVENTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVENTS)
for _MODULE_NAME, _SAVED_MODULE in _SAVED_MODULES.items():
    if _SAVED_MODULE is None:
        del sys.modules[_MODULE_NAME]
    else:
        sys.modules[_MODULE_NAME] = _SAVED_MODULE


class _ArticulationStub:
    def __init__(self):
        self.data = SimpleNamespace(
            default_joint_pos=torch.tensor([[1.0, 0.0, -2.0], [0.5, 1.5, 0.0]]),
            default_joint_vel=torch.zeros(2, 3),
            soft_joint_pos_limits=torch.tensor(
                [[[-3.0, 3.0]] * 3, [[-3.0, 3.0]] * 3], dtype=torch.float
            ),
            soft_joint_vel_limits=torch.full((2, 3), 10.0),
        )
        self.written_state = None

    def write_joint_state_to_sim(self, joint_pos, joint_vel, *, joint_ids, env_ids) -> None:
        self.written_state = (joint_pos.clone(), joint_vel.clone(), joint_ids, env_ids.clone())


class _PhysicsViewStub:
    def __init__(self):
        self.materials = torch.zeros(2, 6, 3)
        self.materials[..., 2] = 0.42
        self.written_env_ids = None

    def get_material_properties(self) -> torch.Tensor:
        return self.materials.clone()

    def set_material_properties(self, materials: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.materials = materials.clone()
        self.written_env_ids = env_ids.clone()


class _WrenchComposerStub:
    def __init__(self):
        self.calls = []

    def set_forces_and_torques(self, **kwargs) -> None:
        self.calls.append(
            {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )


class _PushArticulationStub:
    def __init__(self):
        self.data = SimpleNamespace(
            root_vel_w=torch.zeros(2, 6),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        )
        self.root_physx_view = SimpleNamespace(get_masses=lambda: torch.tensor([[20.0, 30.0], [25.0, 35.0]]))
        self.permanent_wrench_composer = _WrenchComposerStub()
        self.velocity_writes = []

    def write_root_velocity_to_sim(self, root_velocity_w: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.data.root_vel_w[env_ids] = root_velocity_w
        self.velocity_writes.append((root_velocity_w.clone(), env_ids.clone()))


def test_reset_samples_absolute_joint_velocity_independently(monkeypatch) -> None:
    """Zero default velocities must not suppress the requested reset perturbation."""
    position_scale = torch.tensor([[0.9, 1.0, 1.1], [1.1, 0.9, 1.0]])
    sampled_velocity = torch.tensor([[-0.1, 0.0, 0.1], [0.05, -0.05, 0.02]])

    def _sample_uniform(lower, upper, shape, device):
        if (lower, upper) == (0.9, 1.1):
            return position_scale.to(device).reshape(shape)
        if (lower, upper) == (-0.1, 0.1):
            return sampled_velocity.to(device).reshape(shape)
        raise AssertionError(f"Unexpected range: {(lower, upper)}")

    monkeypatch.setattr(_EVENTS.math_utils, "sample_uniform", _sample_uniform, raising=False)
    asset = _ArticulationStub()
    env = SimpleNamespace(scene={"robot": asset})
    env_ids = torch.tensor([0, 1])

    _EVENTS.reset_joints_by_position_scale_and_velocity(
        env,
        env_ids,
        position_range=(0.9, 1.1),
        velocity_range=(-0.1, 0.1),
    )

    assert asset.written_state is not None
    joint_pos, joint_vel, joint_ids, written_env_ids = asset.written_state
    torch.testing.assert_close(joint_pos, asset.data.default_joint_pos * position_scale)
    torch.testing.assert_close(joint_vel, sampled_velocity)
    assert joint_ids == slice(None)
    torch.testing.assert_close(written_env_ids, env_ids)


def test_correlated_material_buckets_preserve_friction_ratio(monkeypatch) -> None:
    """Every material bucket must keep dynamic friction at 75% of static friction."""
    samples = iter((torch.tensor([0.5, 0.7, 0.9]), torch.tensor([0.1, 0.2, 0.3])))
    monkeypatch.setattr(
        _EVENTS.math_utils,
        "sample_uniform",
        lambda lower, upper, shape, device: next(samples).to(device).reshape(shape),
        raising=False,
    )

    buckets = _EVENTS._sample_correlated_material_buckets((0.5, 0.9), 0.75, (0.1, 0.3), 3)

    torch.testing.assert_close(buckets[:, 0], torch.tensor([0.5, 0.7, 0.9]))
    torch.testing.assert_close(buckets[:, 1], buckets[:, 0] * 0.75)
    torch.testing.assert_close(buckets[:, 2], torch.tensor([0.1, 0.2, 0.3]))


def test_correlated_material_event_uses_one_bucket_per_environment(monkeypatch) -> None:
    """Non-foot shapes use nominal friction while both feet share one randomized material."""
    event = object.__new__(_EVENTS.randomize_rigid_body_material_correlated)
    event.asset_cfg = SimpleNamespace(body_ids=[1, 2])
    event.num_shapes_per_body = [2, 2, 2]
    event.material_buckets = torch.tensor([[0.5, 0.375, 0.1], [0.9, 0.675, 0.3]])
    event.asset = SimpleNamespace(root_physx_view=_PhysicsViewStub())
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=2))
    monkeypatch.setattr(torch, "randint", lambda low, high, shape, device: torch.tensor([0, 1]))

    event(
        env,
        torch.tensor([0, 1]),
        default_static_friction=0.8,
        default_dynamic_friction=0.6,
        static_friction_range=(0.5, 0.9),
        dynamic_friction_ratio=0.75,
        restitution_range=(0.1, 0.3),
        num_buckets=2,
        asset_cfg=_SceneEntityCfgStub("robot"),
    )

    expected_non_foot = torch.tensor([0.8, 0.6, 0.42]).expand(2, 3)
    expected_env_0_feet = torch.tensor([0.5, 0.375, 0.1]).expand(4, 3)
    expected_env_1_feet = torch.tensor([0.9, 0.675, 0.3]).expand(4, 3)
    torch.testing.assert_close(event.asset.root_physx_view.materials[0, :2], expected_non_foot)
    torch.testing.assert_close(event.asset.root_physx_view.materials[1, :2], expected_non_foot)
    torch.testing.assert_close(event.asset.root_physx_view.materials[0, 2:], expected_env_0_feet)
    torch.testing.assert_close(event.asset.root_physx_view.materials[1, 2:], expected_env_1_feet)


def test_force_from_delta_velocity_matches_requested_impulse() -> None:
    """The force pulse must integrate to mass times the requested velocity change."""
    delta_velocity_w = torch.tensor([[0.5, -0.25, 0.0], [-0.2, 0.4, 0.0]])
    total_mass = torch.tensor([60.0, 75.0])
    duration_s = torch.tensor([0.1, 0.25])

    force_w = _EVENTS._force_from_delta_velocity(delta_velocity_w, total_mass, duration_s)

    torch.testing.assert_close(force_w * duration_s[:, None], total_mass[:, None] * delta_velocity_w)


def test_mixed_push_selects_one_mode_and_clears_force(monkeypatch) -> None:
    """One event uses velocity or force, and a force pulse is cleared when its timer expires."""
    asset_cfg = _SceneEntityCfgStub("robot")
    asset_cfg.body_ids = [0]
    params = {
        "asset_cfg": asset_cfg,
        "push_interval_range_s": (10.0, 15.0),
        "delta_velocity_x_range": (-0.5, 1.0),
        "delta_velocity_y_range": (-0.5, 0.5),
        "velocity_push_probability": 0.5,
        "force_duration_range_s": (0.05, 0.5),
    }
    cfg = SimpleNamespace(params=params)
    asset = _PushArticulationStub()
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.01, scene={"robot": asset})
    event = _EVENTS.RoK4MixedPush(cfg, env)
    event._push_time_left_s[:] = 0.0
    event._total_mass = torch.tensor([50.0, 60.0])

    def _sample_uniform(value_range, count):
        if value_range == (-0.5, 1.0):
            return torch.tensor([0.2, -0.4])[:count]
        if value_range == (-0.5, 0.5):
            return torch.tensor([0.1, 0.3])[:count]
        if value_range == (10.0, 15.0):
            return torch.full((count,), 12.0)
        raise AssertionError(f"Unexpected range: {value_range}")

    monkeypatch.setattr(event, "_sample_uniform", _sample_uniform)
    monkeypatch.setattr(event, "_sample_force_duration", lambda value_range, count, dt: torch.full((count,), 0.1))
    monkeypatch.setattr(_EVENTS.math_utils, "yaw_quat", lambda quat: quat, raising=False)
    monkeypatch.setattr(_EVENTS.math_utils, "quat_apply", lambda quat, vector: vector, raising=False)
    monkeypatch.setattr(_EVENTS.torch, "rand", lambda count, device: torch.tensor([0.25, 0.75])[:count])

    event(env, torch.tensor([0, 1]), **params)

    torch.testing.assert_close(asset.data.root_vel_w[0, :3], torch.tensor([0.2, 0.1, 0.0]))
    torch.testing.assert_close(asset.data.root_vel_w[1, :3], torch.zeros(3))
    torch.testing.assert_close(event._force_w[1], torch.tensor([-240.0, 180.0, 0.0]))
    torch.testing.assert_close(event._force_time_left_s, torch.tensor([0.0, 0.1]))
    assert len(asset.velocity_writes) == 1
    assert len(asset.permanent_wrench_composer.calls) == 1

    event._push_time_left_s[1] = 5.0
    event._force_time_left_s[1] = env.step_dt
    event(env, torch.tensor([1]), **params)

    torch.testing.assert_close(event._force_w[1], torch.zeros(3))
    torch.testing.assert_close(asset.permanent_wrench_composer.calls[-1]["forces"], torch.zeros(1, 1, 3))
