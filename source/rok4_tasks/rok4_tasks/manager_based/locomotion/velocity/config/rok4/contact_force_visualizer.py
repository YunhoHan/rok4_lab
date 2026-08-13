# Copyright (c) 2026, RoK4 Lab Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4 contact-force debug visualization."""

from __future__ import annotations

from collections import deque

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis


ROK4_CONTACT_FORCE_ARROW_BASE_SCALE = 0.08
"""Uniform prototype scale used as the contact-force arrow thickness."""


ROK4_CONTACT_FORCE_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/RoK4ContactForces",
    markers={
        "left_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(ROK4_CONTACT_FORCE_ARROW_BASE_SCALE,) * 3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
        ),
        "right_force": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(ROK4_CONTACT_FORCE_ARROW_BASE_SCALE,) * 3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2)),
        ),
        "hidden": sim_utils.SphereCfg(radius=0.001, visible=False),
    },
)
"""Marker prototypes for left/right RoK4 foot contact-force vectors."""


class RoK4ContactForceVisualizer(ContactSensor):
    """Contact sensor with env-0 total ground-reaction-force visualization."""

    _FOOT_BODY_NAMES = ("L_Foot_Link", "R_Foot_Link")
    _ARROW_BASE_SCALE = ROK4_CONTACT_FORCE_ARROW_BASE_SCALE
    _FORCE_TO_LENGTH_SCALE = 1.0e-3
    _MAX_ARROW_LENGTH = 0.75
    _ARROW_TAIL_OFFSET_RATIO = 0.25
    _ARROW_TAIL_CLEARANCE = 0.015
    _PLOT_HISTORY_DURATION = 3.0
    _PLOT_SAMPLE_PERIOD = 0.01
    _PLOT_MAX_DATAPOINTS = round(_PLOT_HISTORY_DURATION / _PLOT_SAMPLE_PERIOD) + 1
    _PLOT_TIME_MINOR_TICK = 0.1
    _PLOT_TIME_MAJOR_TICK = 0.5
    _PLOT_TIME_GRID_DATAPOINTS = 1501
    _PLOT_MIN_FORCE = 0.0
    _PLOT_MAX_FORCE = 4000.0
    _PLOT_FORCE_TICK = 1000.0
    _PLOT_HEIGHT = 240
    _Y_AXIS_WIDTH = 80
    _WINDOW_WIDTH = 600
    _WINDOW_HEIGHT = 650
    _PANEL_MARGIN = 10
    _PANEL_SPACING = 4
    _TITLE_FONT_SIZE = 22
    _LABEL_FONT_SIZE = 18
    _VALUE_FONT_SIZE = 24
    _AXIS_FONT_SIZE = 18

    def __init__(self, cfg):
        """Initialize the contact sensor and debug-visualization state."""
        self._foot_body_ids: list[int] | None = None
        self._force_visualizer: VisualizationMarkers | None = None
        self._force_window = None
        self._left_force_label = None
        self._right_force_label = None
        self._force_plot_data: list[list[float]] = [[], []]
        self._force_grid_plots = []
        self._force_time_grid_plots = []
        self._left_force_plot = None
        self._right_force_plot = None
        self._force_plot_times: deque[float] = deque(maxlen=self._PLOT_MAX_DATAPOINTS)
        self._plot_time_labels = []
        self._force_plot_elapsed_time = 0.0
        self._force_plot_last_sensor_time: float | None = None
        super().__init__(cfg)

    def __del__(self):
        """Destroy the force panel and unsubscribe sensor callbacks."""
        self._destroy_force_window()
        super().__del__()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self._force_visualizer is None:
                self._force_visualizer = VisualizationMarkers(ROK4_CONTACT_FORCE_MARKER_CFG)
            self._force_visualizer.set_visibility(True)
            self._create_force_window()
            if self._force_window is not None:
                self._force_window.visible = True
            self._clear_force_plot()
        else:
            if self._force_visualizer is not None:
                self._force_visualizer.set_visibility(False)
            if self._force_window is not None:
                self._force_window.visible = False

    def _debug_vis_callback(self, event):
        del event
        if self.body_physx_view is None:
            return

        foot_body_ids = self._resolve_foot_body_ids()
        force_vectors = self._ground_reaction_forces_w()
        force_magnitudes = torch.linalg.vector_norm(force_vectors, dim=-1)
        force_directions = self._force_directions(force_vectors, force_magnitudes)
        orientations = self._force_direction_orientations(force_directions)

        transforms = self.body_physx_view.get_transforms().view(self._num_envs, self._num_bodies, 7)
        foot_positions = transforms[0, foot_body_ids, :3]

        arrow_lengths = torch.clamp(
            force_magnitudes * self._FORCE_TO_LENGTH_SCALE,
            max=self._MAX_ARROW_LENGTH,
        )
        # arrow_x.usd spans local x=[-0.25, 0.75]. Shift its origin so the tail starts above the foot.
        tail_offsets = arrow_lengths * self._ARROW_TAIL_OFFSET_RATIO + self._ARROW_TAIL_CLEARANCE
        arrow_origins = foot_positions + force_directions * tail_offsets.unsqueeze(-1)
        scales = torch.ones((2, 3), device=force_vectors.device)
        scales[:, 0] = arrow_lengths / self._ARROW_BASE_SCALE
        marker_indices = torch.tensor([0, 1], dtype=torch.long, device=force_vectors.device)
        marker_indices[force_magnitudes <= self.cfg.force_threshold] = 2

        self._force_visualizer.visualize(
            translations=arrow_origins,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
        )
        self._update_force_panel(force_magnitudes)

    def _ground_reaction_forces_w(self) -> torch.Tensor:
        """Return each foot's total ground contact force [N] in the world frame."""
        if self.data.force_matrix_w is None or self.data.friction_forces_w is None:
            raise RuntimeError(
                "RoK4 contact-force visualization requires a ground contact filter and friction-force tracking."
            )

        foot_ids = self._resolve_foot_body_ids()
        normal_forces_w = self.data.force_matrix_w[0, foot_ids].sum(dim=1)
        tangential_forces_w = self.data.friction_forces_w[0, foot_ids].sum(dim=1)
        return torch.nan_to_num(normal_forces_w + tangential_forces_w)

    def _resolve_foot_body_ids(self) -> list[int]:
        """Resolve and cache the left/right foot body indices."""
        if self._foot_body_ids is None:
            body_ids, body_names = self.find_bodies(self._FOOT_BODY_NAMES, preserve_order=True)
            if body_names != list(self._FOOT_BODY_NAMES):
                raise RuntimeError(
                    "RoK4 contact-force visualization requires L_Foot_Link and R_Foot_Link. "
                    f"Resolved: {body_names}."
                )
            self._foot_body_ids = body_ids
        return self._foot_body_ids

    @staticmethod
    def _force_directions(
        force_vectors: torch.Tensor, force_magnitudes: torch.Tensor
    ) -> torch.Tensor:
        """Return normalized force vectors, falling back to +X for zero force."""
        x_axis = torch.zeros_like(force_vectors)
        x_axis[:, 0] = 1.0
        return torch.where(
            (force_magnitudes > 1.0e-6).unsqueeze(-1),
            force_vectors / force_magnitudes.clamp_min(1.0e-6).unsqueeze(-1),
            x_axis,
        )

    @staticmethod
    def _force_direction_orientations(directions: torch.Tensor) -> torch.Tensor:
        """Return quaternions that rotate each marker's +X axis onto a force direction."""
        x_axis = torch.zeros_like(directions)
        x_axis[:, 0] = 1.0
        rotation_axes = torch.linalg.cross(x_axis, directions, dim=-1)
        axis_norms = torch.linalg.vector_norm(rotation_axes, dim=-1)
        fallback_axis = torch.zeros_like(rotation_axes)
        fallback_axis[:, 2] = 1.0
        rotation_axes = torch.where(
            (axis_norms > 1.0e-6).unsqueeze(-1),
            rotation_axes / axis_norms.clamp_min(1.0e-6).unsqueeze(-1),
            fallback_axis,
        )
        rotation_angles = torch.acos(torch.clamp(directions[:, 0], -1.0, 1.0))
        return quat_from_angle_axis(rotation_angles, rotation_axes)

    def _create_force_window(self):
        if self._force_window is not None:
            return

        try:
            import omni.ui as ui
        except ModuleNotFoundError:
            # The headless Kit experience does not load omni.ui. Force arrows can still be rendered offscreen.
            return

        self._force_window = ui.Window(
            "RoK4 Contact Forces",
            width=self._WINDOW_WIDTH,
            height=self._WINDOW_HEIGHT,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._force_window.frame:
            with ui.VStack(spacing=self._PANEL_SPACING, style={"margin": self._PANEL_MARGIN}):
                ui.Label(
                    "Ground reaction |F| [N] (env 0)",
                    height=30,
                    style={"font_size": self._TITLE_FONT_SIZE},
                )
                with ui.HStack(height=30):
                    ui.Label(
                        "Left foot",
                        width=110,
                        style={"color": 0xFFFF6600, "font_size": self._LABEL_FONT_SIZE},
                    )
                    self._left_force_label = ui.Label(
                        "0.0 N",
                        width=130,
                        alignment=ui.Alignment.RIGHT_CENTER,
                        style={"font_size": self._VALUE_FONT_SIZE},
                    )
                    ui.Spacer()
                with ui.HStack(height=30):
                    ui.Label(
                        "Right foot",
                        width=110,
                        style={"color": 0xFF33FF00, "font_size": self._LABEL_FONT_SIZE},
                    )
                    self._right_force_label = ui.Label(
                        "0.0 N",
                        width=130,
                        alignment=ui.Alignment.RIGHT_CENTER,
                        style={"font_size": self._VALUE_FONT_SIZE},
                    )
                    ui.Spacer()
                ui.Label(
                    "Recent env-0 GRF magnitude",
                    height=26,
                    style={"font_size": self._LABEL_FONT_SIZE},
                )
                self._build_force_plot(ui)
                with ui.HStack(height=26):
                    ui.Spacer(width=self._Y_AXIS_WIDTH)
                    ui.Spacer()
                    ui.Label(
                        "Left |F|",
                        width=110,
                        alignment=ui.Alignment.CENTER,
                        style={"color": 0xFFFF6600, "font_size": self._AXIS_FONT_SIZE},
                    )
                    ui.Spacer(width=20)
                    ui.Label(
                        "Right |F|",
                        width=110,
                        alignment=ui.Alignment.CENTER,
                        style={"color": 0xFF33FF00, "font_size": self._AXIS_FONT_SIZE},
                    )
                    ui.Spacer()
                self._build_force_time_axis(ui)
                ui.Label(
                    "Simulation time [s]",
                    height=24,
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": self._AXIS_FONT_SIZE},
                )

    def _build_force_plot(self, ui):
        """Build a fixed-range force graph without filter or autoscale controls."""
        tick_values = tuple(
            float(value)
            for value in range(
                int(self._PLOT_MAX_FORCE),
                int(self._PLOT_MIN_FORCE) - int(self._PLOT_FORCE_TICK),
                -int(self._PLOT_FORCE_TICK),
            )
        )
        # omni.ui.Plot reserves lower widget space. Keep labels in its actual data area.
        tick_spacing = (self._PLOT_HEIGHT - 50) / (len(tick_values) - 1)
        tick_label_height = 26

        with ui.HStack(height=self._PLOT_HEIGHT):
            with ui.ZStack(width=self._Y_AXIS_WIDTH, height=self._PLOT_HEIGHT):
                for index, tick_value in enumerate(tick_values):
                    tick_y = index * tick_spacing
                    label_y = min(
                        max(tick_y - 0.5 * tick_label_height, 0.0),
                        self._PLOT_HEIGHT - tick_label_height,
                    )
                    with ui.Placer(offset_y=label_y):
                        ui.Label(
                            f"{tick_value:.0f}",
                            width=self._Y_AXIS_WIDTH - 8,
                            height=tick_label_height,
                            alignment=ui.Alignment.RIGHT_CENTER,
                            style={"font_size": self._AXIS_FONT_SIZE},
                        )

            with ui.ZStack(height=self._PLOT_HEIGHT):
                ui.Rectangle(
                    style={
                        "background_color": 0xFF171717,
                        "border_color": 0xFFB0B0B0,
                        "border_width": 1.0,
                    }
                )
                self._force_grid_plots = []
                for tick_value in tick_values:
                    grid_plot = ui.Plot(
                        ui.Type.LINE,
                        self._PLOT_MIN_FORCE,
                        self._PLOT_MAX_FORCE,
                        tick_value,
                        tick_value,
                        height=self._PLOT_HEIGHT,
                        style={"color": 0xFF505050, "background_color": 0x00000000},
                    )
                    grid_plot.scale_min = self._PLOT_MIN_FORCE
                    grid_plot.scale_max = self._PLOT_MAX_FORCE
                    self._force_grid_plots.append(grid_plot)
                self._left_force_plot = ui.Plot(
                    ui.Type.LINE,
                    self._PLOT_MIN_FORCE,
                    self._PLOT_MAX_FORCE,
                    0.0,
                    height=self._PLOT_HEIGHT,
                    style={"color": 0xFFFF6600, "background_color": 0x00000000},
                )
                self._right_force_plot = ui.Plot(
                    ui.Type.LINE,
                    self._PLOT_MIN_FORCE,
                    self._PLOT_MAX_FORCE,
                    0.0,
                    height=self._PLOT_HEIGHT,
                    style={"color": 0xFF33FF00, "background_color": 0x00000000},
                )
                self._build_force_time_grid(ui)

    def _build_force_time_grid(self, ui):
        """Build vertical 0.1-s minor and 0.5-s major grid lines."""
        minor_interval_count = round(self._PLOT_HISTORY_DURATION / self._PLOT_TIME_MINOR_TICK)
        major_interval_stride = round(self._PLOT_TIME_MAJOR_TICK / self._PLOT_TIME_MINOR_TICK)
        minor_grid = [self._PLOT_MIN_FORCE] * self._PLOT_TIME_GRID_DATAPOINTS
        major_grid = [self._PLOT_MIN_FORCE] * self._PLOT_TIME_GRID_DATAPOINTS
        last_data_index = self._PLOT_TIME_GRID_DATAPOINTS - 1
        for tick_index in range(minor_interval_count + 1):
            data_index = round(tick_index * last_data_index / minor_interval_count)
            grid_data = major_grid if tick_index % major_interval_stride == 0 else minor_grid
            grid_data[data_index] = self._PLOT_MAX_FORCE

        self._force_time_grid_plots = []
        for grid_data, color in ((minor_grid, 0xFF353535), (major_grid, 0xFF686868)):
            grid_plot = ui.Plot(
                ui.Type.HISTOGRAM,
                self._PLOT_MIN_FORCE,
                self._PLOT_MAX_FORCE,
                *grid_data,
                height=self._PLOT_HEIGHT,
                style={"color": color, "background_color": 0x00000000},
            )
            grid_plot.scale_min = self._PLOT_MIN_FORCE
            grid_plot.scale_max = self._PLOT_MAX_FORCE
            self._force_time_grid_plots.append(grid_plot)

    def _build_force_time_axis(self, ui):
        """Build evenly spaced 0.5-s labels below the force plot."""
        major_interval_count = round(self._PLOT_HISTORY_DURATION / self._PLOT_TIME_MAJOR_TICK)
        self._plot_time_labels = []
        with ui.HStack(height=26):
            ui.Spacer(width=self._Y_AXIS_WIDTH)
            with ui.ZStack():
                with ui.HStack(spacing=0):
                    for _ in range(major_interval_count):
                        with ui.ZStack(width=ui.Fraction(1)):
                            label = ui.Label(
                                "0.00",
                                alignment=ui.Alignment.LEFT_CENTER,
                                style={"font_size": self._AXIS_FONT_SIZE},
                            )
                            self._plot_time_labels.append(label)
                self._plot_time_labels.append(
                    ui.Label(
                        "0.00",
                        alignment=ui.Alignment.RIGHT_CENTER,
                        style={"font_size": self._AXIS_FONT_SIZE},
                    )
                )

    def _update_force_panel(self, force_magnitudes: torch.Tensor):
        """Update env-0 contact-force labels and live plot from one host transfer."""
        if (
            self._left_force_label is None
            and self._right_force_label is None
            and self._left_force_plot is None
            and self._right_force_plot is None
        ):
            return
        left_force, right_force = force_magnitudes.detach().cpu().tolist()
        if self._left_force_label is not None and self._right_force_label is not None:
            self._left_force_label.text = f"{left_force:.1f} N"
            self._right_force_label.text = f"{right_force:.1f} N"
        if self._left_force_plot is not None and self._right_force_plot is not None:
            sim_time = self._get_elapsed_sim_time()
            if len(self._force_plot_data[0]) >= self._PLOT_MAX_DATAPOINTS:
                for series in self._force_plot_data:
                    del series[0]
            self._force_plot_times.append(sim_time)
            self._force_plot_data[0].append(left_force)
            self._force_plot_data[1].append(right_force)
            self._left_force_plot.set_data(*self._force_plot_data[0])
            self._right_force_plot.set_data(*self._force_plot_data[1])
            for plot in (self._left_force_plot, self._right_force_plot):
                plot.scale_min = self._PLOT_MIN_FORCE
                plot.scale_max = self._PLOT_MAX_FORCE
            self._update_force_plot_time_labels()

    def _get_elapsed_sim_time(self) -> float:
        """Return monotonic elapsed physics time since the plot was enabled [s]."""
        sensor_time = float(self._timestamp[0].item())
        if self._force_plot_last_sensor_time is None:
            self._force_plot_last_sensor_time = sensor_time
            return self._force_plot_elapsed_time

        sensor_dt = sensor_time - self._force_plot_last_sensor_time
        if sensor_dt < 0.0:
            # Sensor timestamps restart at zero when an environment resets.
            sensor_dt = sensor_time
        self._force_plot_elapsed_time += max(sensor_dt, 0.0)
        self._force_plot_last_sensor_time = sensor_time
        return self._force_plot_elapsed_time

    def _update_force_plot_time_labels(self):
        """Update the 0.5-s labels of the plot time axis [s]."""
        if not self._force_plot_times:
            return
        end_time = self._force_plot_times[-1]
        start_time = max(end_time - self._PLOT_HISTORY_DURATION, 0.0)
        for tick_index, label in enumerate(self._plot_time_labels):
            tick_time = start_time + tick_index * self._PLOT_TIME_MAJOR_TICK
            label.text = f"{tick_time:.2f}"

    def _reset_force_plot_times(self):
        """Clear plot timestamps and reset the visible time-axis labels."""
        self._force_plot_times.clear()
        self._force_plot_elapsed_time = 0.0
        self._force_plot_last_sensor_time = None
        for label in self._plot_time_labels:
            label.text = "0.00"

    def _clear_force_plot(self):
        """Clear force samples, time samples, and visible plot data."""
        for series in self._force_plot_data:
            series.clear()
        if self._left_force_plot is not None:
            self._left_force_plot.set_data()
        if self._right_force_plot is not None:
            self._right_force_plot.set_data()
        self._reset_force_plot_times()

    def _destroy_force_window(self):
        if self._force_window is not None:
            self._force_window.visible = False
            self._force_window.destroy()
            self._force_window = None
        self._left_force_label = None
        self._right_force_label = None
        self._plot_time_labels = []
        self._left_force_plot = None
        self._right_force_plot = None
        self._force_grid_plots = []
        self._force_time_grid_plots = []
        self._clear_force_plot()

    def _invalidate_initialize_callback(self, event):
        self._foot_body_ids = None
        super()._invalidate_initialize_callback(event)
