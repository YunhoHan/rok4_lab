# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the concurrently trained RoK4 base-velocity estimator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict


_PPO_PATH = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/rok4_ppo.py"
_SPEC = importlib.util.spec_from_file_location("rok4_ppo_under_test", _PPO_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PPO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PPO)


def _observations(batch_size: int = 8) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, 240),
            "critic": torch.randn(batch_size, 240),
            "privileged": torch.randn(batch_size, 10),
        },
        batch_size=[batch_size],
    )


def _policy() -> object:
    return _PPO.RoK4EstimatorActorCritic(
        _observations(),
        {"policy": ["policy"], "critic": ["critic", "privileged"]},
        13,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        estimator_hidden_dims=[32, 16],
        activation="elu",
        estimator_activation="elu",
        init_noise_std=1.0,
    )


def test_estimator_and_actor_dimensions_preserve_control_contract() -> None:
    """The estimator uses 225D while the external observation and action stay 240D and 13D."""
    policy = _policy()
    observations = _observations()

    estimator_observation = policy.actor.estimator_observation(observations["policy"])
    estimated_velocity = policy.estimate_base_velocity(observations)
    actions = policy.act_inference(observations)
    critic_value = policy.evaluate(observations)

    assert estimator_observation.shape == (8, 225)
    assert estimated_velocity.shape == (8, 3)
    assert actions.shape == (8, 13)
    assert critic_value.shape == (8, 1)


def test_estimator_excludes_all_command_history_values() -> None:
    """Changing only command history must not alter the estimator input or output."""
    policy = _policy()
    actor_observation = torch.randn(4, 240)
    changed_command = actor_observation.clone()
    changed_command[:, 30:45] += torch.randn(4, 15) * 100.0

    estimator_input = policy.actor.estimator_observation(actor_observation)
    changed_estimator_input = policy.actor.estimator_observation(changed_command)
    estimate = policy.actor.estimate(actor_observation)
    changed_estimate = policy.actor.estimate(changed_command)

    torch.testing.assert_close(changed_estimator_input, estimator_input)
    torch.testing.assert_close(changed_estimate, estimate)


def test_ppo_actor_gradient_is_detached_from_estimator() -> None:
    """PPO action gradients must not update the supervised estimator."""
    policy = _policy()
    actor_observation = torch.randn(4, 240)

    policy.actor(actor_observation).square().mean().backward()

    assert all(parameter.grad is None for parameter in policy.estimator_parameters())
    assert any(parameter.grad is not None for parameter in policy.actor.policy.parameters())


def test_estimator_mse_updates_only_estimator_parameters() -> None:
    """Supervised velocity loss must update the estimator without entering the action MLP."""
    policy = _policy()
    observations = _observations(batch_size=4)

    estimate = policy.estimate_base_velocity(observations)
    target = policy.estimator_target(observations)
    torch.nn.functional.mse_loss(estimate, target).backward()

    assert any(parameter.grad is not None for parameter in policy.estimator_parameters())
    assert all(parameter.grad is None for parameter in policy.actor.policy.parameters())


def test_torchscript_exporter_accepts_only_original_240d_observation() -> None:
    """The backward-compatible TorchScript module must infer velocity and emit 13 actions."""
    policy = _policy().eval()
    exporter = _PPO._RoK4PolicyExporter(policy, policy.actor_obs_normalizer).eval()

    actions = exporter(torch.randn(1, 240))

    assert actions.shape == (1, 13)
    assert exporter.actor.policy[0].in_features == 243


def test_fused_exporter_is_torchscript_compatible() -> None:
    """The fused estimator and action policy must script as one deployment graph."""
    policy = _policy().eval()
    exporter = _PPO._RoK4PolicyExporter(policy, policy.actor_obs_normalizer).eval()

    scripted = torch.jit.script(exporter)
    actions = scripted(torch.randn(1, 240))

    assert actions.shape == (1, 13)


def test_onnx_exporter_returns_action_and_the_exact_internal_estimate() -> None:
    """The diagnostic output must be the same estimate used to compute the action."""
    policy = _policy().eval()
    action_exporter = _PPO._RoK4PolicyExporter(policy, policy.actor_obs_normalizer).eval()
    onnx_exporter = _PPO._RoK4OnnxPolicyExporter(policy, policy.actor_obs_normalizer).eval()
    observation = torch.randn(1, 240)

    expected_actions = action_exporter(observation)
    actions, estimated_velocity = onnx_exporter(observation)
    normalized_observation = onnx_exporter.normalizer(observation)
    expected_velocity = onnx_exporter.actor.estimate(normalized_observation).detach()

    assert actions.shape == (1, 13)
    assert estimated_velocity.shape == (1, 3)
    torch.testing.assert_close(actions, expected_actions)
    torch.testing.assert_close(estimated_velocity, expected_velocity)


def test_onnx_file_has_two_named_outputs_without_changing_actions(tmp_path: Path) -> None:
    """The exported graph must expose separate 13D action and 3D velocity tensors."""
    onnx = pytest.importorskip("onnx")
    from onnx.reference import ReferenceEvaluator

    policy = _policy().eval()
    observation = torch.randn(1, 240)

    _PPO.export_rok4_policy_as_onnx(
        policy,
        str(tmp_path),
        normalizer=policy.actor_obs_normalizer,
    )

    model_path = tmp_path / "policy.onnx"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    assert [output.name for output in model.graph.output] == ["actions", "estimated_base_lin_vel_b"]

    evaluator = ReferenceEvaluator(model)
    actions, estimated_velocity = evaluator.run(None, {"obs": observation.numpy()})
    expected_actions, expected_velocity = _PPO._RoK4OnnxPolicyExporter(
        policy,
        policy.actor_obs_normalizer,
    ).eval()(observation)

    assert actions.shape == (1, 13)
    assert estimated_velocity.shape == (1, 3)
    np.testing.assert_allclose(actions, expected_actions.detach().numpy(), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(
        estimated_velocity,
        expected_velocity.detach().numpy(),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_ppo_update_optimizes_estimator_and_reports_physical_rmse() -> None:
    """One PPO update must run both disjoint optimizers and expose velocity RMSE."""
    policy = _policy()
    observations = _observations(batch_size=8)
    algorithm = _PPO.RoK4PPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        device="cpu",
    )
    with torch.no_grad():
        actions = policy.act(observations)
        old_log_probability = policy.get_actions_log_prob(actions).unsqueeze(-1)
        old_mean = policy.action_mean.clone()
        old_std = policy.action_std.clone()
        values = policy.evaluate(observations)

    class _Storage:
        cleared = False

        def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
            assert num_mini_batches == 1
            assert num_epochs == 1
            yield (
                observations,
                actions,
                values,
                torch.randn(8, 1),
                values + torch.randn(8, 1),
                old_log_probability,
                old_mean,
                old_std,
                (None, None),
                None,
            )

        def clear(self) -> None:
            self.cleared = True

    storage = _Storage()
    algorithm.storage = storage
    estimator_before = [parameter.detach().clone() for parameter in policy.estimator_parameters()]

    losses = algorithm.update()

    assert storage.cleared
    assert "estimator_mse" in losses
    assert set(algorithm.estimator_metrics) == {"rmse_vx", "rmse_vy", "rmse_vz", "rmse_total"}
    assert any(
        not torch.equal(before, after)
        for before, after in zip(estimator_before, policy.estimator_parameters(), strict=True)
    )
