"""Actuator-space actions for RoK4 velocity-locomotion tasks."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from rok4_tasks.assets.robots.rok4_adapt import RoK4AdaptActuator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RoK4ActuatorPositionAction(ActionTerm):
    r"""Interpret policy output as normalized RoK4 actuator-position offsets.

    The clipped raw action is scaled in actuator coordinates, added to the
    default actuator pose, and mapped through ADAPT to a joint-position target.
    """

    cfg: RoK4ActuatorPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: RoK4ActuatorPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        if cfg.raw_action_clip[0] >= cfg.raw_action_clip[1]:
            raise ValueError(f"raw_action_clip must satisfy min < max, received {cfg.raw_action_clip}.")

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) != 13:
            raise ValueError(f"RoK4 actuator action requires 13 joints, resolved {len(self._joint_ids)}.")

        actuator = self._asset.actuators.get(cfg.actuator_name)
        if not isinstance(actuator, RoK4AdaptActuator):
            raise TypeError(
                f"Asset actuator '{cfg.actuator_name}' must be RoK4AdaptActuator, received {type(actuator).__name__}."
            )
        if set(self._joint_names) != set(actuator.joint_names):
            raise ValueError(
                "Action joints and ADAPT actuator joints differ.\n"
                f"Action: {self._joint_names}\nActuator: {actuator.joint_names}"
            )

        self._transmission = actuator.transmission
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._scaled_actuator_actions = torch.zeros_like(self._raw_actions)
        self._actuator_position_targets = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        indices, _, values = string_utils.resolve_matching_names_values(cfg.scale, self._joint_names)
        if len(indices) != self.action_dim:
            raise ValueError("RoK4 actuator action scale must resolve exactly one value for every actuator.")
        self._scale = torch.zeros(self.action_dim, device=self.device)
        self._scale[indices] = torch.tensor(values, dtype=torch.float, device=self.device)

        default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._default_actuator_pos = self._transmission.joint_to_actuator_position(default_joint_pos)
        self._print_adapt_startup_diagnostics(default_joint_pos)
        self.reset()

    @property
    def action_dim(self) -> int:
        """Number of actuator commands."""
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Clipped raw actuator actions in the normalized policy range."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Mapped joint-position targets [rad] sent to the articulation."""
        return self._processed_actions

    @property
    def scaled_actuator_actions(self) -> torch.Tensor:
        """Actuator-position offsets after action scaling [rad]."""
        return self._scaled_actuator_actions

    @property
    def actuator_position_targets(self) -> torch.Tensor:
        """Absolute actuator-position targets [rad]."""
        return self._actuator_position_targets

    @property
    def default_actuator_pos(self) -> torch.Tensor:
        """Default actuator positions converted from the asset's default joint pose [rad]."""
        return self._default_actuator_pos

    def process_actions(self, actions: torch.Tensor) -> None:
        """Clip and convert normalized actuator actions to joint targets."""
        self._raw_actions[:] = torch.clamp(actions, min=self.cfg.raw_action_clip[0], max=self.cfg.raw_action_clip[1])
        self._scaled_actuator_actions[:] = self._raw_actions * self._scale
        self._actuator_position_targets[:] = self._default_actuator_pos + self._scaled_actuator_actions
        self._processed_actions[:] = self._transmission.actuator_to_joint_position(self._actuator_position_targets)

    def apply_actions(self) -> None:
        """Write the mapped joint target for actuator-space PD processing."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        """Reset action buffers to the default actuator and joint pose."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._scaled_actuator_actions[env_ids] = 0.0
        self._actuator_position_targets[env_ids] = self._default_actuator_pos[env_ids]
        self._processed_actions[env_ids] = self._transmission.actuator_to_joint_position(
            self._actuator_position_targets[env_ids]
        )

    def _print_adapt_startup_diagnostics(self, default_joint_pos: torch.Tensor) -> None:
        """Validate and print ADAPT matrices and the default-pose conversion."""
        transmission = self._transmission
        q_default = default_joint_pos[0]
        psi_default = self._default_actuator_pos[0]
        q_default_round_trip = transmission.actuator_to_joint_position(psi_default.unsqueeze(0))[0]

        # Isaac Lab enables CUDA TF32 for training. Validate the matrix identities in CPU FP64 so
        # TF32 rounding does not produce a false structural failure, then check the runtime mapping
        # separately with a tolerance appropriate for CUDA FP32/TF32.
        q_j_psi = transmission.q_j_psi.detach().to(device="cpu", dtype=torch.float64)
        q_j_psi_inv = transmission.q_j_psi_inv.detach().to(device="cpu", dtype=torch.float64)
        q_j_psi_t = transmission.q_j_psi_t.detach().to(device="cpu", dtype=torch.float64)
        q_j_psi_inv_t = transmission.q_j_psi_inv_t.detach().to(device="cpu", dtype=torch.float64)
        identity = torch.eye(4, dtype=torch.float64)
        inverse_right_error = torch.max(torch.abs(q_j_psi @ q_j_psi_inv - identity)).item()
        inverse_left_error = torch.max(torch.abs(q_j_psi_inv @ q_j_psi - identity)).item()
        transpose_error = torch.max(torch.abs(q_j_psi_t - q_j_psi.T)).item()
        inverse_transpose_error = torch.max(torch.abs(q_j_psi_inv_t - q_j_psi_inv.T)).item()

        q_default_fp64 = q_default.detach().to(device="cpu", dtype=torch.float64)
        psi_default_fp64 = transmission.joint_to_actuator_position(q_default_fp64.unsqueeze(0))[0]
        q_default_round_trip_fp64 = transmission.actuator_to_joint_position(psi_default_fp64.unsqueeze(0))[0]
        reference_round_trip_error = torch.max(torch.abs(q_default_round_trip_fp64 - q_default_fp64)).item()
        runtime_mapping_error = torch.max(
            torch.abs(psi_default.detach().to(device="cpu", dtype=torch.float64) - psi_default_fp64)
        ).item()
        runtime_round_trip_error = torch.max(torch.abs(q_default_round_trip - q_default)).item()

        print("\n[RoK4 ADAPT] Transmission matrices")
        self._print_matrix("q_J_psi", transmission.q_j_psi)
        self._print_matrix("q_J_psi_inv", transmission.q_j_psi_inv)
        self._print_matrix("q_J_psi_T", transmission.q_j_psi_t)
        self._print_matrix("q_J_psi_invT", transmission.q_j_psi_inv_t)
        print("[RoK4 ADAPT] Validation errors")
        print(f"  max|J @ J^-1 - I| (FP64)       = {inverse_right_error:.3e}")
        print(f"  max|J^-1 @ J - I| (FP64)       = {inverse_left_error:.3e}")
        print(f"  max|stored J^T - J^T| (FP64)   = {transpose_error:.3e}")
        print(f"  max|stored J^-T - J^-T| (FP64) = {inverse_transpose_error:.3e}")
        print(f"  max|q0 FP64 round trip - q0|    = {reference_round_trip_error:.3e}")
        print(f"  max|psi0 runtime - psi0 FP64|   = {runtime_mapping_error:.3e}")
        print(f"  max|q0 runtime round trip - q0| = {runtime_round_trip_error:.3e}")

        print("[RoK4 ADAPT] Default pose conversion: q_default (13x1) -> psi_default (13x1)")
        header = f"{'idx':>3} | {'coordinate':<25} | {'q_default [rad]':>16} | {'psi_default [rad]':>18}"
        print(header)
        print("-" * len(header))
        for index, (name, joint_value, actuator_value) in enumerate(
            zip(
                self._joint_names,
                q_default.detach().cpu().tolist(),
                psi_default.detach().cpu().tolist(),
                strict=True,
            )
        ):
            print(f"{index:>3} | {name:<25} | {joint_value:>16.8f} | {actuator_value:>18.8f}")

        structural_tolerance = 1.0e-5
        structural_error = max(
            inverse_right_error,
            inverse_left_error,
            transpose_error,
            inverse_transpose_error,
            reference_round_trip_error,
        )
        if structural_error > structural_tolerance:
            raise RuntimeError(
                "RoK4 ADAPT structural validation failed: "
                f"maximum CPU FP64 error {structural_error:.3e} exceeds {structural_tolerance:.1e}."
            )

        runtime_tolerance = 5.0e-4
        runtime_error = max(runtime_mapping_error, runtime_round_trip_error)
        if runtime_error > runtime_tolerance:
            raise RuntimeError(
                "RoK4 ADAPT runtime validation failed: "
                f"maximum CUDA mapping error {runtime_error:.3e} exceeds {runtime_tolerance:.1e}."
            )

    @staticmethod
    def _print_matrix(name: str, matrix: torch.Tensor) -> None:
        """Print a small matrix with stable numeric formatting."""
        print(f"{name} =")
        for row in matrix.detach().cpu().tolist():
            print("  [" + ", ".join(f"{value: .8f}" for value in row) + "]")


@configclass
class RoK4ActuatorPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`RoK4ActuatorPositionAction`."""

    class_type: type[ActionTerm] = RoK4ActuatorPositionAction
    asset_name: str = "robot"

    joint_names: list[str] = MISSING
    """Exact joint order corresponding to the 13 actuator coordinates."""

    actuator_name: str = "body"
    """Name of the ADAPT actuator group in the articulation configuration."""

    scale: dict[str, float] = MISSING
    """Raw action to actuator-position offset scales [rad]."""

    raw_action_clip: tuple[float, float] = (-1.0, 1.0)
    """Clip range applied before actuator scaling."""
