"""Left-right symmetry augmentation for RoK4 actuator-space locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


_POLICY_HISTORY_LENGTH = 5
_VECTOR_DIM = 3
_ACTUATOR_DIM = 13

# Isaac Lab flattens each term's history before concatenating the terms. The policy layout is therefore
# [base_ang_vel(5x3), projected_gravity(5x3), command(5x3), actuator_pos(5x13),
#  actuator_vel(5x13), last_action(5x13)], rather than five contiguous 48-value frames.
_POLICY_OBS_DIM = _POLICY_HISTORY_LENGTH * (3 * _VECTOR_DIM + 3 * _ACTUATOR_DIM)

_ACTUATOR_MIRROR_INDICES = (6, 7, 8, 9, 11, 10, 0, 1, 2, 3, 5, 4, 12)
_ACTUATOR_MIRROR_SIGNS = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append one left-right mirrored copy of each RoK4 PPO sample.

    The returned batch contains the original samples first and their mirrored
    counterparts second. This ordering matches RSL-RL's symmetry data
    augmentation path, which repeats the original advantages, returns, value
    targets, and old action log probabilities for the mirrored half.

    Args:
        env: Environment required by the RSL-RL augmentation callback API.
        obs: Observation groups to augment.
        actions: Raw actuator-space policy actions to augment.

    Returns:
        The doubled observation and action batches. Either result is ``None``
        when the corresponding input is ``None``.
    """
    del env

    if obs is not None:
        unexpected_groups = set(obs.keys()) - {"policy", "privileged"}
        if unexpected_groups:
            raise ValueError(f"RoK4 symmetry does not define transforms for observation groups: {unexpected_groups}.")

        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _mirror_policy_observation(obs["policy"])

        if "privileged" in obs.keys():
            obs_aug["privileged"][:batch_size] = obs["privileged"]
            obs_aug["privileged"][batch_size:] = _mirror_privileged_observation(obs["privileged"])
    else:
        obs_aug = None

    if actions is not None:
        actions_aug = torch.cat((actions, _mirror_actuator_coordinates(actions)), dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _mirror_policy_observation(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the five-history, 240-value RoK4 policy observation."""
    if obs.shape[-1] != _POLICY_OBS_DIM:
        raise ValueError(f"Expected {_POLICY_OBS_DIM} policy values, received shape {tuple(obs.shape)}.")

    mirrored = torch.empty_like(obs)
    offset = 0

    vector_term_size = _POLICY_HISTORY_LENGTH * _VECTOR_DIM
    actuator_term_size = _POLICY_HISTORY_LENGTH * _ACTUATOR_DIM

    # Angular velocity is an axial vector under reflection across the robot x-z plane.
    mirrored[..., offset : offset + vector_term_size] = _mirror_vector_history(
        obs[..., offset : offset + vector_term_size], (-1.0, 1.0, -1.0)
    )
    offset += vector_term_size

    # Projected gravity is a polar vector.
    mirrored[..., offset : offset + vector_term_size] = _mirror_vector_history(
        obs[..., offset : offset + vector_term_size], (1.0, -1.0, 1.0)
    )
    offset += vector_term_size

    # Direct command layout is [vx, vy, wz].
    mirrored[..., offset : offset + vector_term_size] = _mirror_vector_history(
        obs[..., offset : offset + vector_term_size], (1.0, -1.0, -1.0)
    )
    offset += vector_term_size

    for _ in range(3):
        values = obs[..., offset : offset + actuator_term_size]
        values = values.reshape(*values.shape[:-1], _POLICY_HISTORY_LENGTH, _ACTUATOR_DIM)
        mirrored_values = _mirror_actuator_coordinates(values)
        mirrored[..., offset : offset + actuator_term_size] = mirrored_values.flatten(start_dim=-2)
        offset += actuator_term_size

    return mirrored


def _mirror_privileged_observation(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the critic-only current base linear velocity ``[vx, vy, vz]``."""
    if obs.shape[-1] != _VECTOR_DIM:
        raise ValueError(f"Expected {_VECTOR_DIM} privileged values, received shape {tuple(obs.shape)}.")
    signs = obs.new_tensor((1.0, -1.0, 1.0))
    return obs * signs


def _mirror_vector_history(values: torch.Tensor, signs: tuple[float, float, float]) -> torch.Tensor:
    """Mirror one flattened five-sample vector history without reversing time."""
    history = values.reshape(*values.shape[:-1], _POLICY_HISTORY_LENGTH, _VECTOR_DIM)
    return (history * history.new_tensor(signs)).flatten(start_dim=-2)


def _mirror_actuator_coordinates(values: torch.Tensor) -> torch.Tensor:
    """Swap left/right actuator coordinates and apply reflection signs.

    The coupled ADAPT coordinates use ``[psi_1, psi_2, psi_3, psi_4]`` per
    leg. Swapping ``psi_3`` and ``psi_4`` mirrors joint ankle roll while
    preserving joint ankle pitch.
    """
    if values.shape[-1] != _ACTUATOR_DIM:
        raise ValueError(f"Expected {_ACTUATOR_DIM} actuator values, received shape {tuple(values.shape)}.")
    indices = torch.tensor(_ACTUATOR_MIRROR_INDICES, device=values.device)
    signs = values.new_tensor(_ACTUATOR_MIRROR_SIGNS)
    return values[..., indices] * signs
