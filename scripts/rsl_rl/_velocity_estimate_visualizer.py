# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation visualization for the RoK4 base-velocity estimate."""

from __future__ import annotations

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul


ROK4_ESTIMATED_VELOCITY_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/velocity_estimated",
    markers={
        "arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
        )
    },
)
"""Orange arrow marker for estimated planar base velocity."""


class RoK4EstimatedVelocityVisualizer:
    """Draw estimated body-frame planar base velocity above the robot."""

    _HEIGHT_OFFSET = 0.58
    _VELOCITY_TO_LENGTH_SCALE = 3.0

    def __init__(self, velocity_command) -> None:
        """Initialize the visualizer from the active base-velocity command term.

        Args:
            velocity_command: Command term that owns the robot articulation used
                by the current and target velocity arrows.
        """
        if not hasattr(velocity_command, "robot"):
            raise TypeError("Estimated velocity visualization requires a velocity command with a robot asset.")
        self._velocity_command = velocity_command
        self._visualizer = VisualizationMarkers(ROK4_ESTIMATED_VELOCITY_MARKER_CFG)
        self._visualizer.set_visibility(True)

    def visualize(self, estimated_base_lin_vel_b: torch.Tensor) -> None:
        """Draw estimated base-frame ``vx`` and ``vy`` [m/s].

        Args:
            estimated_base_lin_vel_b: Estimated body-frame linear velocity
                ``[vx, vy, vz]`` [m/s], shape ``(num_envs, 3)``.
        """
        if estimated_base_lin_vel_b.ndim != 2 or estimated_base_lin_vel_b.shape[1] != 3:
            raise ValueError(
                "Expected estimated base velocity with shape (num_envs, 3), received "
                f"{tuple(estimated_base_lin_vel_b.shape)}."
            )

        robot = self._velocity_command.robot
        base_pos_w = self._as_torch(robot.data.root_pos_w).clone()
        base_quat_w = self._as_torch(robot.data.root_quat_w)
        if estimated_base_lin_vel_b.shape[0] != base_pos_w.shape[0]:
            raise ValueError(
                "Estimated velocity environment count does not match the robot state: "
                f"{estimated_base_lin_vel_b.shape[0]} != {base_pos_w.shape[0]}."
            )

        planar_velocity_b = estimated_base_lin_vel_b[:, :2].to(device=base_pos_w.device)
        arrow_scale, arrow_quat_w = self._resolve_planar_velocity(planar_velocity_b, base_quat_w)
        base_pos_w[:, 2] += self._HEIGHT_OFFSET
        self._visualizer.visualize(base_pos_w, arrow_quat_w, arrow_scale)

    @classmethod
    def _resolve_planar_velocity(
        cls,
        planar_velocity_b: torch.Tensor,
        base_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve body-frame planar velocity into world-frame arrow transforms."""
        default_scale = ROK4_ESTIMATED_VELOCITY_MARKER_CFG.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=planar_velocity_b.device).repeat(
            planar_velocity_b.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.vector_norm(planar_velocity_b, dim=1)
        arrow_scale[:, 0] *= cls._VELOCITY_TO_LENGTH_SCALE

        heading_angle = torch.atan2(planar_velocity_b[:, 1], planar_velocity_b[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat_b = quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, quat_mul(base_quat_w, arrow_quat_b)

    @staticmethod
    def _as_torch(value) -> torch.Tensor:
        """Return an Isaac Lab tensor or Warp array as a Torch tensor."""
        if isinstance(value, torch.Tensor):
            return value
        return wp.to_torch(value)
