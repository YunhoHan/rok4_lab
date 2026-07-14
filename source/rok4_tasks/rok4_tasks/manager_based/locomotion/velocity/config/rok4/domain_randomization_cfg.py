"""Domain randomization settings for RoK4 locomotion tasks."""

from copy import deepcopy

ROK4_FOOT_BODY_NAMES = ["L_Foot_Link", "R_Foot_Link"]
ROK4_BASE_BODY_NAMES = ["Base_Link"]
ROK4_UPPER_BODY_NAMES = ["Upper_Body_Link"]
ROK4_LOWER_BODY_NAMES = [
    "L_Hip_Yaw_Link",
    "L_Hip_Roll_Link",
    "L_Thigh_Link",
    "L_Calf_Link",
    "L_Ankle_Pitch_Link",
    "L_Ankle_Roll_Link",
    "L_Foot_Link",
    "R_Hip_Yaw_Link",
    "R_Hip_Roll_Link",
    "R_Thigh_Link",
    "R_Calf_Link",
    "R_Ankle_Pitch_Link",
    "R_Ankle_Roll_Link",
    "R_Foot_Link",
]

# Startup DR is sampled per environment when the scene is created.
# These terms are relatively expensive in large-vectorized training, so keep them at startup by default.
ROK4_PHYSICS_MATERIAL_MODE = "startup"
ROK4_STATIC_FRICTION_RANGE = (0.8, 0.8)
ROK4_DYNAMIC_FRICTION_RANGE = (0.6, 0.6)
ROK4_RESTITUTION_RANGE = (0.1, 0.3)
ROK4_MATERIAL_BUCKETS = 64

ROK4_BASE_MASS_MODE = "startup"
ROK4_BASE_MASS_SCALE_RANGE = (0.9, 1.1)
ROK4_BASE_MASS_OPERATION = "scale"
ROK4_UPPER_MASS_MODE = "startup"
ROK4_UPPER_MASS_SCALE_RANGE = (0.9, 1.25)
ROK4_UPPER_MASS_OPERATION = "scale"
ROK4_LOWER_MASS_MODE = "startup"
ROK4_LOWER_MASS_SCALE_RANGE = (0.9, 1.25)
ROK4_LOWER_MASS_OPERATION = "scale"

ROK4_BASE_COM_MODE = "startup"
ROK4_BASE_COM_RANGE = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}
ROK4_UPPER_COM_MODE = "startup"
ROK4_UPPER_COM_RANGE = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}
ROK4_LOWER_COM_MODE = "startup"
ROK4_LOWER_COM_RANGE = {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)}

# Reset DR is sampled again every episode.
ROK4_EXTERNAL_WRENCH_MODE = "reset"
ROK4_EXTERNAL_FORCE_RANGE = (0.0, 0.0)
ROK4_EXTERNAL_TORQUE_RANGE = (-0.0, 0.0)

# Reset initial-state randomization.
ROK4_RESET_JOINT_POSITION_RANGE = (0.9, 1.1)
ROK4_RESET_BASE_POSE_RANGE = {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)}
ROK4_RESET_BASE_VELOCITY_RANGE = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.1, 0.1),
}


def _configure_mass_randomization(
    event_cfg,
    body_names: list[str],
    mode: str,
    scale_range: tuple[float, float],
    operation: str,
) -> None:
    """Configure one rigid-body mass randomization event."""
    event_cfg.mode = mode
    event_cfg.params["asset_cfg"].body_names = list(body_names)
    event_cfg.params["mass_distribution_params"] = scale_range
    event_cfg.params["operation"] = operation


def _configure_com_randomization(
    event_cfg,
    body_names: list[str],
    mode: str,
    com_range: dict[str, tuple[float, float]],
) -> None:
    """Configure one rigid-body COM randomization event."""
    event_cfg.mode = mode
    event_cfg.params["asset_cfg"].body_names = list(body_names)
    event_cfg.params["com_range"] = dict(com_range)


def apply_rok4_domain_randomization(env_cfg) -> None:
    """Apply RoK4 domain randomization settings to an environment config."""
    # Startup DR: sampled per environment when the scene is created.
    env_cfg.events.physics_material.mode = ROK4_PHYSICS_MATERIAL_MODE
    env_cfg.events.physics_material.params["asset_cfg"].body_names = list(ROK4_FOOT_BODY_NAMES)
    env_cfg.events.physics_material.params["static_friction_range"] = ROK4_STATIC_FRICTION_RANGE
    env_cfg.events.physics_material.params["dynamic_friction_range"] = ROK4_DYNAMIC_FRICTION_RANGE
    env_cfg.events.physics_material.params["restitution_range"] = ROK4_RESTITUTION_RANGE
    env_cfg.events.physics_material.params["num_buckets"] = ROK4_MATERIAL_BUCKETS

    # Mass DR is split into base, upper-body, and lower-body groups so each range can be tuned separately.
    _configure_mass_randomization(
        env_cfg.events.add_base_mass,
        ROK4_BASE_BODY_NAMES,
        ROK4_BASE_MASS_MODE,
        ROK4_BASE_MASS_SCALE_RANGE,
        ROK4_BASE_MASS_OPERATION,
    )
    env_cfg.events.add_upper_mass = deepcopy(env_cfg.events.add_base_mass)
    _configure_mass_randomization(
        env_cfg.events.add_upper_mass,
        ROK4_UPPER_BODY_NAMES,
        ROK4_UPPER_MASS_MODE,
        ROK4_UPPER_MASS_SCALE_RANGE,
        ROK4_UPPER_MASS_OPERATION,
    )
    env_cfg.events.add_lower_mass = deepcopy(env_cfg.events.add_base_mass)
    _configure_mass_randomization(
        env_cfg.events.add_lower_mass,
        ROK4_LOWER_BODY_NAMES,
        ROK4_LOWER_MASS_MODE,
        ROK4_LOWER_MASS_SCALE_RANGE,
        ROK4_LOWER_MASS_OPERATION,
    )

    # COM DR is also split by body group. Lower-body COM uses a smaller range because the links are small.
    _configure_com_randomization(env_cfg.events.base_com, ROK4_BASE_BODY_NAMES, ROK4_BASE_COM_MODE, ROK4_BASE_COM_RANGE)
    env_cfg.events.upper_com = deepcopy(env_cfg.events.base_com)
    _configure_com_randomization(env_cfg.events.upper_com, ROK4_UPPER_BODY_NAMES, ROK4_UPPER_COM_MODE, ROK4_UPPER_COM_RANGE)
    env_cfg.events.lower_com = deepcopy(env_cfg.events.base_com)
    _configure_com_randomization(env_cfg.events.lower_com, ROK4_LOWER_BODY_NAMES, ROK4_LOWER_COM_MODE, ROK4_LOWER_COM_RANGE)

    # Reset DR: sampled again every episode.
    env_cfg.events.base_external_force_torque.mode = ROK4_EXTERNAL_WRENCH_MODE
    env_cfg.events.base_external_force_torque.params["asset_cfg"].body_names = list(ROK4_BASE_BODY_NAMES)
    env_cfg.events.base_external_force_torque.params["force_range"] = ROK4_EXTERNAL_FORCE_RANGE
    env_cfg.events.base_external_force_torque.params["torque_range"] = ROK4_EXTERNAL_TORQUE_RANGE

    # Reset initial-state randomization.
    env_cfg.events.reset_robot_joints.params["position_range"] = ROK4_RESET_JOINT_POSITION_RANGE
    env_cfg.events.reset_base.params = {
        "pose_range": dict(ROK4_RESET_BASE_POSE_RANGE),
        "velocity_range": dict(ROK4_RESET_BASE_VELOCITY_RANGE),
    }
