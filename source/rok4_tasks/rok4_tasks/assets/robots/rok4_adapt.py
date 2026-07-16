"""ADAPT transmission and actuator model for RoK4."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class RoK4AdaptTransmission:
    r"""Linear RoK4 ADAPT transmission between actuator and joint coordinates.

    The coupled leg coordinates satisfy ``q = J psi``. Hip yaw, hip roll, and
    torso yaw are direct-drive coordinates and therefore pass through unchanged.
    """

    NUM_COORDINATES = 13
    COUPLED_SLICES = (slice(2, 6), slice(8, 12))

    def __init__(
        self,
        link_alpha: float = 0.09845,
        link_beta: float = 0.06,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float,
    ) -> None:
        """Initialize the constant transmission matrices.

        Args:
            link_alpha: ADAPT reference link length [m].
            link_beta: ADAPT differential link length [m].
            device: Torch device on which matrices are allocated.
            dtype: Torch floating-point data type used by the matrices.
        """
        if link_alpha <= 0.0 or link_beta <= 0.0:
            raise ValueError("ADAPT link lengths must be positive.")

        self.link_alpha = float(link_alpha)
        self.link_beta = float(link_beta)
        self.ratio = self.link_beta / self.link_alpha

        self.q_j_psi = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.5, -0.5, 0.0, 0.0],
                [-0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, -self.ratio, self.ratio],
            ],
            device=device,
            dtype=dtype,
        )
        self.q_j_psi_inv = torch.linalg.inv(self.q_j_psi)
        self.q_j_psi_t = self.q_j_psi.T
        self.q_j_psi_inv_t = self.q_j_psi_inv.T

    def actuator_to_joint_position(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """Map actuator positions ``psi`` [rad] to joint positions ``q`` [rad]."""
        return self._map_coupled(actuator_pos, self.q_j_psi.T)

    def joint_to_actuator_position(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Map joint positions ``q`` [rad] to actuator positions ``psi`` [rad]."""
        return self._map_coupled(joint_pos, self.q_j_psi_inv.T)

    def actuator_to_joint_velocity(self, actuator_vel: torch.Tensor) -> torch.Tensor:
        """Map actuator velocities [rad/s] to joint velocities [rad/s]."""
        return self.actuator_to_joint_position(actuator_vel)

    def joint_to_actuator_velocity(self, joint_vel: torch.Tensor) -> torch.Tensor:
        """Map joint velocities [rad/s] to actuator velocities [rad/s]."""
        return self.joint_to_actuator_position(joint_vel)

    def actuator_to_joint_acceleration(self, actuator_acc: torch.Tensor) -> torch.Tensor:
        """Map actuator accelerations [rad/s^2] to joint accelerations [rad/s^2]."""
        return self.actuator_to_joint_position(actuator_acc)

    def joint_to_actuator_acceleration(self, joint_acc: torch.Tensor) -> torch.Tensor:
        """Map joint accelerations [rad/s^2] to actuator accelerations [rad/s^2]."""
        return self.joint_to_actuator_position(joint_acc)

    def actuator_to_joint_torque(self, actuator_torque: torch.Tensor) -> torch.Tensor:
        r"""Map actuator torques to joint torques using ``tau_q = J^-T tau_psi`` [N m]."""
        return self._map_coupled(actuator_torque, self.q_j_psi_inv)

    def joint_to_actuator_torque(self, joint_torque: torch.Tensor) -> torch.Tensor:
        r"""Map joint torques to actuator torques using ``tau_psi = J^T tau_q`` [N m]."""
        return self._map_coupled(joint_torque, self.q_j_psi)

    def _map_coupled(self, values: torch.Tensor, row_matrix: torch.Tensor) -> torch.Tensor:
        """Apply a row-vector transform to both four-coordinate ADAPT blocks."""
        if values.shape[-1] != self.NUM_COORDINATES:
            raise ValueError(
                f"RoK4 ADAPT expects {self.NUM_COORDINATES} coordinates, received shape {tuple(values.shape)}."
            )

        matrix = row_matrix.to(device=values.device, dtype=values.dtype)
        mapped = values.clone()
        for coupled_slice in self.COUPLED_SLICES:
            mapped[..., coupled_slice] = values[..., coupled_slice] @ matrix
        return mapped


class RoK4AdaptActuator(IdealPDActuator):
    r"""Explicit actuator-space PD model for the RoK4 ADAPT transmission.

    Joint targets from Isaac Lab are converted to actuator targets. PD torques are
    computed and clipped in actuator coordinates before being mapped back to the
    joint efforts sent to PhysX.
    """

    cfg: RoK4AdaptActuatorCfg

    def __init__(self, cfg: RoK4AdaptActuatorCfg, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

        if self.num_joints != RoK4AdaptTransmission.NUM_COORDINATES:
            raise ValueError(f"RoK4 ADAPT actuator requires 13 joints, received {self.num_joints}.")
        if set(self.joint_names) != set(cfg.expected_joint_names):
            raise ValueError(
                "RoK4 actuator joints do not match the configured ADAPT joints.\n"
                f"Resolved: {self.joint_names}\nExpected: {cfg.expected_joint_names}"
            )
        if not 0.0 < cfg.torque_limit_factor <= 1.0:
            raise ValueError("torque_limit_factor must be in (0, 1].")
        if not 0.0 < cfg.velocity_limit_factor <= 1.0:
            raise ValueError("velocity_limit_factor must be in (0, 1].")

        self.transmission = RoK4AdaptTransmission(
            link_alpha=cfg.link_alpha,
            link_beta=cfg.link_beta,
            device=self._device,
        )
        # Isaac Lab actuator groups follow USD storage order. ADAPT calculations use the
        # left-leg, right-leg, torso canonical order configured in expected_joint_names.
        self._canonical_ids_in_model = torch.tensor(
            [self.joint_names.index(name) for name in cfg.expected_joint_names],
            dtype=torch.long,
            device=self._device,
        )
        self.actuator_torque_limit_max = self._limit_tensor(
            cfg.actuator_torque_limit, "actuator_torque_limit"
        )
        self.actuator_velocity_limit_max = self._limit_tensor(
            cfg.actuator_velocity_limit, "actuator_velocity_limit"
        )
        self.actuator_torque_limit = self.actuator_torque_limit_max * cfg.torque_limit_factor
        self.actuator_velocity_limit = self.actuator_velocity_limit_max * cfg.velocity_limit_factor

        # Isaac Lab's standard buffers follow USD model order; custom actuator buffers remain canonical.
        self.effort_limit = self._canonical_to_model(self.actuator_torque_limit)
        self.velocity_limit = self._canonical_to_model(self.actuator_velocity_limit)

        self.computed_actuator_effort = torch.zeros_like(self.computed_effort)
        self.applied_actuator_effort = torch.zeros_like(self.computed_effort)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute actuator PD torque and return its joint-space equivalent."""
        if control_action.joint_positions is None:
            raise ValueError("RoK4 ADAPT actuator requires joint-position targets.")

        joint_pos_canonical = self._model_to_canonical(joint_pos)
        joint_pos_target_canonical = self._model_to_canonical(control_action.joint_positions)
        actuator_pos = self.transmission.joint_to_actuator_position(joint_pos_canonical)
        actuator_pos_target = self.transmission.joint_to_actuator_position(joint_pos_target_canonical)

        joint_vel_canonical = self._model_to_canonical(joint_vel)
        actuator_vel = self.transmission.joint_to_actuator_velocity(joint_vel_canonical)
        if control_action.joint_velocities is None:
            actuator_vel_target = torch.zeros_like(actuator_vel)
        else:
            joint_vel_target_canonical = self._model_to_canonical(control_action.joint_velocities)
            actuator_vel_target = self.transmission.joint_to_actuator_velocity(joint_vel_target_canonical)

        if control_action.joint_efforts is None:
            actuator_effort_ff = torch.zeros_like(actuator_pos)
        else:
            joint_effort_ff_canonical = self._model_to_canonical(control_action.joint_efforts)
            actuator_effort_ff = self.transmission.joint_to_actuator_torque(joint_effort_ff_canonical)

        stiffness = self._model_to_canonical(self.stiffness)
        damping = self._model_to_canonical(self.damping)

        self.computed_actuator_effort[:] = (
            stiffness * (actuator_pos_target - actuator_pos)
            + damping * (actuator_vel_target - actuator_vel)
            + actuator_effort_ff
        )
        self.applied_actuator_effort[:] = torch.clamp(
            self.computed_actuator_effort,
            min=-self.actuator_torque_limit,
            max=self.actuator_torque_limit,
        )

        computed_joint_effort = self.transmission.actuator_to_joint_torque(self.computed_actuator_effort)
        applied_joint_effort = self.transmission.actuator_to_joint_torque(self.applied_actuator_effort)
        self.computed_effort[:] = self._canonical_to_model(computed_joint_effort)
        self.applied_effort[:] = self._canonical_to_model(applied_joint_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _limit_tensor(self, values: list[float], name: str) -> torch.Tensor:
        """Convert an ordered actuator limit list to a batched tensor."""
        if len(values) != self.num_joints:
            raise ValueError(f"{name} must contain {self.num_joints} values, received {len(values)}.")
        limit = torch.tensor(values, dtype=torch.float, device=self._device)
        if torch.any(limit <= 0.0):
            raise ValueError(f"{name} values must be positive.")
        return limit.unsqueeze(0).repeat(self._num_envs, 1)

    def _model_to_canonical(self, values: torch.Tensor) -> torch.Tensor:
        """Reorder a tensor from USD model order to canonical ADAPT order."""
        return values[..., self._canonical_ids_in_model]

    def _canonical_to_model(self, values: torch.Tensor) -> torch.Tensor:
        """Reorder a tensor from canonical ADAPT order to USD model order."""
        model_values = torch.empty_like(values)
        model_values[..., self._canonical_ids_in_model] = values
        return model_values


@configclass
class RoK4AdaptActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`RoK4AdaptActuator`."""

    class_type: type = RoK4AdaptActuator

    expected_joint_names: list[str] = MISSING
    """Exact 13-joint order used by the ADAPT matrices."""

    link_alpha: float = 0.09845
    """ADAPT reference link length [m]."""

    link_beta: float = 0.06
    """ADAPT differential link length [m]."""

    actuator_torque_limit: list[float] = MISSING
    """Maximum actuator torques before safety factors [N m]."""

    actuator_velocity_limit: list[float] = MISSING
    """Maximum actuator velocities before safety factors [rad/s]."""

    torque_limit_factor: float = 0.9
    """Factor applied to maximum actuator torque limits."""

    velocity_limit_factor: float = 0.9
    """Factor applied to maximum actuator velocity limits."""
