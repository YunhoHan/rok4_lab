# Copyright (c) 2026, RoK4 Lab Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4 PLAY/TELEOP command monitoring and manual push controls."""

from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RoK4PushTestWindow(ManagerBasedRLEnvWindow):
    """Isaac Lab environment window with command monitoring and manual push controls."""

    def __init__(self, env: ManagerBasedRLEnv, window_name: str = "IsaacLab"):
        """Create the standard RL window and append RoK4 command and push panels.

        Args:
            env: RoK4 manager-based RL environment.
            window_name: Isaac Lab window title.
        """
        self._pending_push_delta_w: tuple[float, float] | None = None
        self._command_status_label = None
        self._next_command_display_time = 0.0
        self._push_magnitude_model = None
        self._push_status_label = None
        super().__init__(env, window_name)
        self._build_command_monitor_frame()
        self._build_push_test_frame()

    def _build_command_monitor_frame(self) -> None:
        """Build a live display of the selected environment's base-frame command."""
        import isaacsim.gui.components.ui_utils as ui_utils
        import omni.ui as ui

        with self.ui_window_elements["main_vstack"]:
            self.ui_window_elements["rok4_command_frame"] = ui.CollapsableFrame(
                title="RoK4 Velocity Monitor",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=ui_utils.get_style(),
            )
            with self.ui_window_elements["rok4_command_frame"]:
                with ui.VStack(spacing=4, height=0):
                    self._command_status_label = ui.Label(
                        "Waiting for velocity data...",
                        height=154,
                    )

    def update_command_display(self) -> None:
        """Refresh the selected environment's commanded velocity at 20 Hz."""
        if self._command_status_label is None:
            return

        current_time = time.monotonic()
        if current_time < self._next_command_display_time:
            return
        self._next_command_display_time = current_time + 0.05

        env_index = self._selected_env_index()
        command = self.env.command_manager.get_command("base_velocity")[env_index, :3]
        robot = self.env.scene["robot"]
        root_quat_w = robot.data.root_quat_w[env_index : env_index + 1]
        root_lin_vel_w = robot.data.root_lin_vel_w[env_index : env_index + 1]
        root_lin_vel_yaw = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w)[0]
        root_ang_vel_w = robot.data.root_ang_vel_w[env_index]

        velocity_data = torch.cat((command, root_lin_vel_yaw, root_ang_vel_w[2:3]))
        command_vx, command_vy, command_wz, actual_vx, actual_vy, actual_vz, actual_wz = (
            velocity_data.detach().to(device="cpu", dtype=torch.float32).tolist()
        )
        command_planar_speed = math.hypot(command_vx, command_vy)
        actual_planar_speed = math.hypot(actual_vx, actual_vy)
        self._command_status_label.text = (
            f"env {env_index}\n"
            "COMMAND [yaw frame]\n"
            f"vx={command_vx:+.3f} | vy={command_vy:+.3f} m/s\n"
            f"wz={command_wz:+.3f} rad/s | |vxy|={command_planar_speed:.3f} m/s\n"
            "ACTUAL [yaw frame]\n"
            f"vx={actual_vx:+.3f} | vy={actual_vy:+.3f} | vz={actual_vz:+.3f} m/s\n"
            f"wz_world={actual_wz:+.3f} rad/s | |vxy|={actual_planar_speed:.3f} m/s"
        )

    def _build_push_test_frame(self) -> None:
        """Build controls for root linear-velocity changes [m/s] in the world frame."""
        import isaacsim.gui.components.ui_utils as ui_utils
        import omni.ui as ui

        with self.ui_window_elements["main_vstack"]:
            self.ui_window_elements["rok4_push_frame"] = ui.CollapsableFrame(
                title="RoK4 Push Test",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=ui_utils.get_style(),
            )
            with self.ui_window_elements["rok4_push_frame"]:
                with ui.VStack(spacing=5, height=0):
                    self._push_magnitude_model = ui_utils.float_builder(
                        label="Delta velocity [m/s]",
                        default_val=0.5,
                        min=0.05,
                        max=1.0,
                        step=0.05,
                        format="%.2f",
                        tooltip="World-frame root linear-velocity change applied at the next policy step.",
                    )
                    with ui.HStack(height=28, spacing=4):
                        ui.Button(
                            "+X",
                            clicked_fn=lambda: self._queue_directional_push(1.0, 0.0),
                            tooltip="Add +X world-frame root velocity.",
                            style=ui_utils.get_style(),
                        )
                        ui.Button(
                            "-X",
                            clicked_fn=lambda: self._queue_directional_push(-1.0, 0.0),
                            tooltip="Add -X world-frame root velocity.",
                            style=ui_utils.get_style(),
                        )
                        ui.Button(
                            "+Y",
                            clicked_fn=lambda: self._queue_directional_push(0.0, 1.0),
                            tooltip="Add +Y world-frame root velocity.",
                            style=ui_utils.get_style(),
                        )
                        ui.Button(
                            "-Y",
                            clicked_fn=lambda: self._queue_directional_push(0.0, -1.0),
                            tooltip="Add -Y world-frame root velocity.",
                            style=ui_utils.get_style(),
                        )
                    ui.Button(
                        "Random XY",
                        height=28,
                        clicked_fn=self._queue_random_push,
                        tooltip="Sample independent world-frame X/Y velocity changes from the selected magnitude.",
                        style=ui_utils.get_style(),
                    )
                    self._push_status_label = ui.Label("Ready", height=22)

    def _queue_directional_push(self, direction_x: float, direction_y: float) -> None:
        """Queue a directional root velocity change [m/s] for the selected environment."""
        magnitude = self._push_magnitude()
        self._queue_push(direction_x * magnitude, direction_y * magnitude)

    def _queue_random_push(self) -> None:
        """Queue a random root velocity change [m/s] matching the training push distribution."""
        magnitude = self._push_magnitude()
        self._queue_push(random.uniform(-magnitude, magnitude), random.uniform(-magnitude, magnitude))

    def _queue_push(self, delta_x: float, delta_y: float) -> None:
        """Store one world-frame root velocity change [m/s] for the inference loop."""
        self._pending_push_delta_w = (delta_x, delta_y)
        if self._push_status_label is not None:
            self._push_status_label.text = f"Queued: dV=({delta_x:+.2f}, {delta_y:+.2f}) m/s"

    def _push_magnitude(self) -> float:
        """Return the clamped velocity-change magnitude [m/s]."""
        if self._push_magnitude_model is None:
            return 0.5
        return min(max(self._push_magnitude_model.as_float, 0.05), 1.0)

    def _selected_env_index(self) -> int:
        """Return the valid environment index currently selected by the viewer."""
        camera_controller = self.env.viewport_camera_controller
        env_index = int(
            camera_controller.cfg.env_index if camera_controller is not None else self.env.cfg.viewer.env_index
        )
        return min(max(env_index, 0), self.env.num_envs - 1)

    def apply_pending_push(self) -> bool:
        """Apply a queued push to the selected environment at a policy-step boundary.

        Returns:
            True when a queued root velocity change was applied, otherwise False.
        """
        if self._pending_push_delta_w is None:
            return False

        delta_x, delta_y = self._pending_push_delta_w
        self._pending_push_delta_w = None
        env_index = self._selected_env_index()
        env_ids = torch.tensor([env_index], dtype=torch.long, device=self.env.device)
        robot = self.env.scene["robot"]
        root_velocity_w = robot.data.root_vel_w[env_ids].clone()
        root_velocity_w[:, 0] += delta_x
        root_velocity_w[:, 1] += delta_y
        robot.write_root_velocity_to_sim(root_velocity_w, env_ids=env_ids)

        if self._push_status_label is not None:
            self._push_status_label.text = (
                f"Applied env {env_index}: dV=({delta_x:+.2f}, {delta_y:+.2f}) m/s"
            )
        return True
