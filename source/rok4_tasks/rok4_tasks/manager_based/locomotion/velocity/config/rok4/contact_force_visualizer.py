# Copyright (c) 2026, RoK4 Lab Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4 contact-force debug visualization."""

from __future__ import annotations

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

    def __init__(self, cfg):
        """Initialize the contact sensor and debug-visualization state."""
        self._foot_body_ids: list[int] | None = None
        self._force_visualizer: VisualizationMarkers | None = None
        self._force_window = None
        self._left_force_label = None
        self._right_force_label = None
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
        self._update_force_labels(force_magnitudes)

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
            width=260,
            height=120,
            visible=True,
            dock_preference=ui.DockPreference.RIGHT_TOP,
        )
        with self._force_window.frame:
            with ui.VStack(spacing=6, style={"margin": 8}):
                ui.Label("Ground reaction |F| [N] (env 0)", height=22)
                with ui.HStack(height=22):
                    ui.Label("Left foot", width=110, style={"color": 0xFFFF6600})
                    self._left_force_label = ui.Label("0.0 N", alignment=ui.Alignment.RIGHT_CENTER)
                with ui.HStack(height=22):
                    ui.Label("Right foot", width=110, style={"color": 0xFF33FF00})
                    self._right_force_label = ui.Label("0.0 N", alignment=ui.Alignment.RIGHT_CENTER)

    def _update_force_labels(self, force_magnitudes: torch.Tensor):
        if self._left_force_label is None or self._right_force_label is None:
            return
        left_force, right_force = force_magnitudes.detach().cpu().tolist()
        self._left_force_label.text = f"{left_force:.1f} N"
        self._right_force_label.text = f"{right_force:.1f} N"

    def _destroy_force_window(self):
        if self._force_window is not None:
            self._force_window.visible = False
            self._force_window.destroy()
            self._force_window = None

    def _invalidate_initialize_callback(self, event):
        self._foot_body_ids = None
        super()._invalidate_initialize_callback(event)
