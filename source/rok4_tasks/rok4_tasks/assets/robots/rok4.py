"""RoK4 robot asset configurations."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

ROK4_LAB_ROOT = Path(__file__).resolve().parents[5]
ROK4_ASSET_DIR = ROK4_LAB_ROOT / "assets" / "rok4_wholebody" / "urdf"

ROK4_JOINT_ORDER = [
    "L_Hip_Yaw_Joint",
    "L_Hip_Roll_Joint",
    "L_Hip_Pitch_Joint",
    "L_Knee_Pitch_Joint",
    "L_Ankle_Pitch_Joint",
    "L_Ankle_Roll_Joint",
    "R_Hip_Yaw_Joint",
    "R_Hip_Roll_Joint",
    "R_Hip_Pitch_Joint",
    "R_Knee_Pitch_Joint",
    "R_Ankle_Pitch_Joint",
    "R_Ankle_Roll_Joint",
    "Torso_Yaw_Joint",
]


def _make_joint_dict(values: list[float]) -> dict[str, float]:
    """Map ordered RoK4 joint values to joint-name dictionaries."""
    return dict(zip(ROK4_JOINT_ORDER, values, strict=True))


# Matches the active Isaac Gym RoK4 defaultJointAngles and norminalJointAngles gait-ready pose.
_ROK4_INIT_JOINT_POS = {
    "Torso_Yaw_Joint": 0.0,
    ".*_Hip_Yaw_Joint": 0.0,
    ".*_Hip_Roll_Joint": 0.0,
    ".*_Hip_Pitch_Joint": -0.0924,
    ".*_Knee_Pitch_Joint": 0.345,
    ".*_Ankle_Pitch_Joint": -0.253,
    ".*_Ankle_Roll_Joint": 0.0,
}

ROK4_KP = _make_joint_dict(
    [
        200.0,
        200.0,
        200.0,
        200.0,
        20.0,
        20.0,
        200.0,
        200.0,
        200.0,
        200.0,
        20.0,
        20.0,
        100.0,
    ]
)
"""Joint position gains used by the explicit torque PD actuator."""

ROK4_KD = _make_joint_dict(
    [
        5.0,
        5.0,
        5.0,
        5.0,
        2.0,
        2.0,
        5.0,
        5.0,
        5.0,
        5.0,
        2.0,
        2.0,
        5.0,
    ]
)
"""Joint velocity gains used by the explicit torque PD actuator."""

ROK4_EFFORT_LIMITS = _make_joint_dict(
    [
        150.0,
        150.0,
        300.0,
        480.0,
        180.0,
        180.0,
        150.0,
        150.0,
        300.0,
        480.0,
        180.0,
        180.0,
        150.0,
    ]
)

ROK4_ARMATURE = _make_joint_dict(
    [
        0.087396415395941,
        0.087396415395941,
        0.17479283079,
        0.17479283079,
        0.084040769086151,
        0.084040769086151,
        0.087396415395941,
        0.087396415395941,
        0.17479283079,
        0.17479283079,
        0.084040769086151,
        0.084040769086151,
        0.087396415395941,
    ]
)

ROK4_STATIC_FRICTION = _make_joint_dict(
    [
        0.226537899603487,
        0.226537899603487,
        0.4530757992,
        3.39806849405,
        1.79822954112,
        1.79822954112,
        0.226537899603487,
        0.226537899603487,
        0.4530757992,
        3.39806849405,
        1.79822954112,
        1.79822954112,
        0.226537899603487,
    ]
)

ROK4_VISCOUS_FRICTION = _make_joint_dict(
    [
        0.250538652440252,
        0.250538652440252,
        0.50107730488,
        0.50107730488,
        0.130617712993805,
        0.130617712993805,
        0.250538652440252,
        0.250538652440252,
        0.50107730488,
        0.50107730488,
        0.130617712993805,
        0.130617712993805,
        0.250538652440252,
    ]
)

ROK4_ACTION_SCALE = _make_joint_dict(
    [
        0.4,
        0.5,
        1.25,
        1.5,
        0.75,
        0.75,
        0.4,
        0.5,
        1.25,
        1.5,
        0.75,
        0.75,
        0.4,
    ]
)
"""Joint action scale values matching the Isaac Gym RoK4 actuator action ranges."""


ROK4_TRAIN_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROK4_ASSET_DIR / "rok4_train.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Isaac Gym references [m]: the straight-leg standing base height is z=0.919.
        # Gait-ready CoM (0.0575 / 2, 0, 0.835) maps to the gait-ready base (0.0552, 0, 0.907).
        # Spawn z=0.929 adds 0.010 m ground clearance above the straight-leg standing base height.
        pos=(0.0552, 0.0, 0.929),
        joint_pos=_ROK4_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "body": IdealPDActuatorCfg(
            joint_names_expr=ROK4_JOINT_ORDER,
            # With IdealPDActuatorCfg, stiffness/damping are explicit torque-PD gains.
            # PhysX position-drive stiffness/damping stay disabled for current-control style simulation.
            effort_limit=ROK4_EFFORT_LIMITS,
            effort_limit_sim=ROK4_EFFORT_LIMITS,
            stiffness=ROK4_KP,
            damping=ROK4_KD,
            armature=ROK4_ARMATURE,
            friction=ROK4_STATIC_FRICTION,
            viscous_friction=ROK4_VISCOUS_FRICTION,
        ),
    },
)
"""RoK4 training asset with primitive visuals/collisions."""


ROK4_TEST_CFG = ROK4_TRAIN_CFG.copy()
ROK4_TEST_CFG.spawn.usd_path = str(ROK4_ASSET_DIR / "rok4_test.usd")
"""RoK4 visual inspection asset with mesh visuals."""
