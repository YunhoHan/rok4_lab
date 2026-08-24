# Copyright (c) 2026, RoK4 Lab Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4 PLAY/TELEOP command monitoring and manual push controls."""

from __future__ import annotations

import asyncio
import math
import random
import time
import weakref
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

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
        self._pending_push_delta_b: tuple[float, float] | None = None
        self._command_status_label = None
        self._next_command_display_time = 0.0
        self._push_magnitude_model = None
        self._push_status_label = None
        self._window_menu_items = []
        super().__init__(env, window_name)
        self._build_command_monitor_frame()
        self._build_push_test_frame()
        self._register_window_menu()
        self.ui_window.set_visibility_changed_fn(self._on_window_visibility_changed)

    def __del__(self):
        """Remove the reopen menu before destroying the Isaac Lab window."""
        try:
            if self.ui_window is not None:
                self.ui_window.set_visibility_changed_fn(None)
            self._remove_window_menu()
        except Exception:
            pass
        super().__del__()

    def _register_window_menu(self) -> None:
        """Add a ``Window > IsaacLab`` entry that restores and re-docks the panel."""
        from omni.kit.menu.utils import MenuItemDescription, add_menu_items

        window_proxy = weakref.proxy(self)
        self._window_menu_items = [
            MenuItemDescription(
                name="IsaacLab",
                onclick_fn=lambda *_args, proxy=window_proxy: proxy._show_and_dock_window(),
            )
        ]
        add_menu_items(self._window_menu_items, "Window")

    def _remove_window_menu(self) -> None:
        """Remove the custom window menu entry if it was registered."""
        if not self._window_menu_items:
            return
        from omni.kit.menu.utils import remove_menu_items

        remove_menu_items(self._window_menu_items, "Window")
        self._window_menu_items = []

    def _show_and_dock_window(self) -> None:
        """Show the existing panel and attach it to the Property dock."""
        if self.ui_window is None:
            return
        was_visible = self.ui_window.visible
        self.ui_window.visible = True
        if was_visible:
            asyncio.ensure_future(self._dock_window(window_title=self.ui_window.title))

    def _on_window_visibility_changed(self, visible: bool) -> None:
        """Restore the standard right-side dock whenever the panel is shown."""
        if visible and self.ui_window is not None:
            asyncio.ensure_future(self._dock_window(window_title=self.ui_window.title))

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
        """Build controls for root linear-velocity changes [m/s] in the base-yaw frame."""
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
                        tooltip="Base-yaw-frame root linear-velocity change applied at the next policy step.",
                    )
                    with ui.HStack(height=28, spacing=4):
                        ui.Button(
                            "+X",
                            clicked_fn=lambda: self._queue_directional_push(1.0, 0.0),
                            tooltip="Push forward along the robot's current base +X direction.",
                            style=ui_utils.get_style(),
                        )
                        ui.Button(
                            "-X",
                            clicked_fn=lambda: self._queue_directional_push(-1.0, 0.0),
                            tooltip="Push backward along the robot's current base -X direction.",
                            style=ui_utils.get_style(),
                        )
                        ui.Button(
                            "+Y",
                            clicked_fn=lambda: self._queue_directional_push(0.0, 1.0),
                            tooltip="Push left along the robot's current base +Y direction.",
                            style=ui_utils.get_style(),
                        )
                        ui.Button(
                            "-Y",
                            clicked_fn=lambda: self._queue_directional_push(0.0, -1.0),
                            tooltip="Push right along the robot's current base -Y direction.",
                            style=ui_utils.get_style(),
                        )
                    ui.Button(
                        "Random XY",
                        height=28,
                        clicked_fn=self._queue_random_push,
                        tooltip="Sample independent base-yaw-frame X/Y velocity changes.",
                        style=ui_utils.get_style(),
                    )
                    self._push_status_label = ui.Label("Ready", height=22)

    def _queue_directional_push(self, direction_x: float, direction_y: float) -> None:
        """Queue a directional root velocity change [m/s] for the selected environment."""
        magnitude = self._push_magnitude()
        self._queue_push(direction_x * magnitude, direction_y * magnitude)

    def _queue_random_push(self) -> None:
        """Queue a random base-yaw-frame root velocity change [m/s]."""
        magnitude = self._push_magnitude()
        self._queue_push(random.uniform(-magnitude, magnitude), random.uniform(-magnitude, magnitude))

    def _queue_push(self, delta_x: float, delta_y: float) -> None:
        """Store one base-yaw-frame root velocity change [m/s] for the inference loop."""
        self._pending_push_delta_b = (delta_x, delta_y)
        if self._push_status_label is not None:
            self._push_status_label.text = f"Queued base: dV=({delta_x:+.2f}, {delta_y:+.2f}) m/s"

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
        if self._pending_push_delta_b is None:
            return False

        delta_x_b, delta_y_b = self._pending_push_delta_b
        self._pending_push_delta_b = None
        env_index = self._selected_env_index()
        env_ids = torch.tensor([env_index], dtype=torch.long, device=self.env.device)
        robot = self.env.scene["robot"]
        root_velocity_w = robot.data.root_vel_w[env_ids].clone()
        delta_velocity_b = root_velocity_w.new_tensor([[delta_x_b, delta_y_b, 0.0]])
        root_quat_w = robot.data.root_quat_w[env_ids]
        delta_velocity_w = quat_apply(yaw_quat(root_quat_w), delta_velocity_b)
        root_velocity_w[:, :3] += delta_velocity_w
        robot.write_root_velocity_to_sim(root_velocity_w, env_ids=env_ids)

        if self._push_status_label is not None:
            self._push_status_label.text = (
                f"Applied env {env_index}: base dV=({delta_x_b:+.2f}, {delta_y_b:+.2f}) m/s"
            )
        return True
