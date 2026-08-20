"""Unit tests for RoK4 left-right symmetry augmentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from tensordict import TensorDict


_SYMMETRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/rok4_tasks/rok4_tasks/manager_based/locomotion/velocity/mdp/symmetry.py"
)
_SPEC = importlib.util.spec_from_file_location("rok4_symmetry_under_test", _SYMMETRY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SYMMETRY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SYMMETRY)


def test_action_mirror_matches_actuator_coordinate_definition() -> None:
    """The actuator action mapping must match the established RoK4 ordering."""
    actions = torch.arange(1, 14, dtype=torch.float).unsqueeze(0)
    mirrored = _SYMMETRY._mirror_actuator_coordinates(actions)
    expected = torch.tensor(
        [[-7.0, -8.0, 9.0, 10.0, 12.0, 11.0, -1.0, -2.0, 3.0, 4.0, 6.0, 5.0, -13.0]]
    )
    torch.testing.assert_close(mirrored, expected)


def test_adapt_actuator_mirror_matches_joint_space_reflection() -> None:
    """Swapping the final actuator pair must negate only joint ankle roll."""
    ratio = 0.06 / 0.09845
    q_j_psi = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, -0.5, 0.0, 0.0],
            [-0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, -ratio, ratio],
        ],
        dtype=torch.float64,
    )
    actuator_mirror = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    joint_mirror = torch.diag(torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float64))
    torch.testing.assert_close(q_j_psi @ actuator_mirror, joint_mirror @ q_j_psi)


def test_observation_and_action_mirrors_are_involutions() -> None:
    """Applying a left-right reflection twice must recover the input."""
    policy = torch.randn(7, 240)
    privileged = torch.randn(7, 10)
    actions = torch.randn(7, 13)
    policy_twice = _SYMMETRY._mirror_policy_observation(_SYMMETRY._mirror_policy_observation(policy))
    privileged_twice = _SYMMETRY._mirror_privileged_observation(
        _SYMMETRY._mirror_privileged_observation(privileged)
    )
    actions_twice = _SYMMETRY._mirror_actuator_coordinates(_SYMMETRY._mirror_actuator_coordinates(actions))
    torch.testing.assert_close(policy_twice, policy)
    torch.testing.assert_close(privileged_twice, privileged)
    torch.testing.assert_close(actions_twice, actions)


def test_policy_mirror_uses_term_major_history_layout() -> None:
    """Every history sample must be mirrored inside its observation term."""
    policy = torch.zeros(1, 240)
    policy[:, 0:15] = torch.tensor([1.0, 2.0, 3.0]).repeat(5)
    policy[:, 15:30] = torch.tensor([4.0, 5.0, 6.0]).repeat(5)
    policy[:, 30:45] = torch.tensor([7.0, 8.0, 9.0]).repeat(5)
    actuator_history = torch.arange(1, 14, dtype=torch.float).repeat(5)
    policy[:, 45:110] = actuator_history
    policy[:, 110:175] = actuator_history
    policy[:, 175:240] = actuator_history

    mirrored = _SYMMETRY._mirror_policy_observation(policy)

    torch.testing.assert_close(mirrored[:, 0:15], torch.tensor([-1.0, 2.0, -3.0]).repeat(5).unsqueeze(0))
    torch.testing.assert_close(mirrored[:, 15:30], torch.tensor([4.0, -5.0, 6.0]).repeat(5).unsqueeze(0))
    torch.testing.assert_close(mirrored[:, 30:45], torch.tensor([7.0, -8.0, -9.0]).repeat(5).unsqueeze(0))
    expected_actuator = torch.tensor(
        [-7.0, -8.0, 9.0, 10.0, 12.0, 11.0, -1.0, -2.0, 3.0, 4.0, 6.0, 5.0, -13.0]
    ).repeat(5)
    for start in (45, 110, 175):
        torch.testing.assert_close(mirrored[:, start : start + 65], expected_actuator.unsqueeze(0))


def test_compute_symmetric_states_doubles_actor_critic_batch() -> None:
    """The RSL-RL callback must preserve originals and append mirrored samples."""
    policy = torch.randn(4, 240)
    critic = torch.randn(4, 240)
    privileged = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
            [31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0],
        ]
    )
    actions = torch.randn(4, 13)
    obs = TensorDict({"policy": policy, "critic": critic, "privileged": privileged}, batch_size=[4])

    obs_aug, actions_aug = _SYMMETRY.compute_symmetric_states(None, obs=obs, actions=actions)

    assert obs_aug is not None
    assert actions_aug is not None
    assert obs_aug.batch_size == torch.Size([8])
    assert actions_aug.shape == (8, 13)
    torch.testing.assert_close(obs_aug["policy"][:4], policy)
    torch.testing.assert_close(obs_aug["policy"][4:], _SYMMETRY._mirror_policy_observation(policy))
    torch.testing.assert_close(obs_aug["critic"][:4], critic)
    torch.testing.assert_close(obs_aug["critic"][4:], _SYMMETRY._mirror_policy_observation(critic))
    torch.testing.assert_close(obs_aug["privileged"][:4], privileged)
    torch.testing.assert_close(
        obs_aug["privileged"][4:],
        _SYMMETRY._mirror_privileged_observation(privileged),
    )
    torch.testing.assert_close(actions_aug[:4], actions)
    torch.testing.assert_close(actions_aug[4:], _SYMMETRY._mirror_actuator_coordinates(actions))


def test_privileged_mirror_reflects_estimator_velocity_target() -> None:
    """Estimator targets must preserve vx/vz and negate body-frame vy."""
    privileged = torch.tensor([[1.2, -0.4, 0.3, 0.907, 0.1, 0.2, 1.0, 0.0, 0.0, 0.5]])

    mirrored = _SYMMETRY._mirror_privileged_observation(privileged)

    torch.testing.assert_close(mirrored[:, :3], torch.tensor([[1.2, 0.4, 0.3]]))
