"""Flat velocity-tracking task for RoK4."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

import rok4_tasks.manager_based.locomotion.velocity.mdp as mdp
from rok4_tasks.assets.robots.rok4 import (
    ROK4_ACTUATOR_ACTION_SCALE,
    ROK4_JOINT_ORDER,
    ROK4_TEST_CFG,
    ROK4_TRAIN_CFG,
)
from rok4_tasks.manager_based.locomotion.velocity.config.rok4.contact_force_visualizer import (
    RoK4ContactForceVisualizer,
)
from rok4_tasks.manager_based.locomotion.velocity.config.rok4.domain_randomization_cfg import (
    apply_rok4_domain_randomization,
)

ROK4_ILLEGAL_CONTACT_BODY_NAMES = [
    "Base_Link",
    "Upper_Body_Link",
    ".*_Hip_Yaw_Link",
    ".*_Hip_Roll_Link",
    ".*_Thigh_Link",
    ".*_Calf_Link",
    ".*_Ankle_Pitch_Link",
    ".*_Ankle_Roll_Link",
]
"""RoK4 body names that must not make contact during flat walking."""

ROK4_LIN_VEL_X_RANGE = (-0.1, 0.85)
"""RoK4 base-frame forward velocity command range [m/s]."""

ROK4_LIN_VEL_Y_RANGE = (-0.3, 0.3)
"""RoK4 base-frame lateral velocity command range [m/s]."""

ROK4_ANG_VEL_Z_RANGE = (-0.6, 0.6)
"""RoK4 yaw velocity command range [rad/s]."""

ROK4_GROUND_COLLISION_PRIM_PATH = "/World/ground/terrain/GroundPlane/CollisionPlane"
"""Collision prim used to isolate foot-ground contact forces in the flat task."""


@configclass
class RoK4ActionsCfg:
    """Actuator-space action specifications for RoK4."""

    actuator_pos = mdp.RoK4ActuatorPositionActionCfg(
        asset_name="robot",
        actuator_name="body",
        joint_names=ROK4_JOINT_ORDER,
        scale=ROK4_ACTUATOR_ACTION_SCALE,
        raw_action_clip=(-1.0, 1.0),
    )


@configclass
class RoK4ObservationsCfg:
    """Blind actuator-space observations for RoK4."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group with the existing 48-value frame shape."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        actuator_pos = ObsTerm(
            func=mdp.actuator_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True),
                "actuator_name": "body",
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        actuator_vel = ObsTerm(
            func=mdp.actuator_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True),
                "actuator_name": "body",
            },
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "actuator_pos"})

        def __post_init__(self):
            """Configure policy history and corruption."""
            self.history_length = 5
            self.flatten_history_dim = True
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RoK4CommandsCfg(CommandsCfg):
    """Direct velocity commands with Gym-style periodic standing windows."""

    base_velocity = mdp.RoK4PeriodicFreezeVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        periodic_freeze_enabled=True,
        periodic_freeze_interval_s=10.0,
        periodic_freeze_duration_range_s=(1.5, 3.0),
        ranges=mdp.RoK4PeriodicFreezeVelocityCommandCfg.Ranges(
            lin_vel_x=ROK4_LIN_VEL_X_RANGE,
            lin_vel_y=ROK4_LIN_VEL_Y_RANGE,
            ang_vel_z=ROK4_ANG_VEL_Z_RANGE,
            heading=None,
        ),
    )


@configclass
class RoK4RewardsCfg(RewardsCfg):
    """Reward terms for RoK4 flat velocity tracking."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["L_Foot_Link", "R_Foot_Link"]),
        },
    )
    feet_flat_orientation_l2 = RewTerm(
        func=mdp.feet_flat_orientation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["L_Foot_Link", "R_Foot_Link"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["L_Foot_Link", "R_Foot_Link"], preserve_order=True
            ),
        },
    )
    feet_stance_width_l2 = None
    stand_still_joint_deviation_l2 = RewTerm(
        func=mdp.stand_still_joint_deviation_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True)},
    )
    joint_action_target_pos_limits = RewTerm(
        func=mdp.joint_action_target_pos_limits,
        weight=-1.0e-3,
        params={
            "action_name": "actuator_pos",
            "asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True),
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw_Joint", ".*_Hip_Roll_Joint"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["Torso_Yaw_Joint"])},
    )
    dof_acc_l2 = None
    dof_torques_l2 = None
    actuator_acc_l2 = RewTerm(
        func=mdp.actuator_acc_l2,
        weight=-1.0e-8,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True)},
    )
    actuator_torques_l2 = RewTerm(
        func=mdp.actuator_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True)},
    )
    actuator_vel_l2 = RewTerm(
        func=mdp.actuator_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True)},
    )
    actuator_velocity_limits = RewTerm(
        func=mdp.actuator_velocity_limits,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True)},
    )
    actuator_torque_limits = RewTerm(
        func=mdp.actuator_torque_limits,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ROK4_JOINT_ORDER, preserve_order=True)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    second_action_rate_l2 = RewTerm(func=mdp.second_action_rate_l2, weight=-0.0005)


@configclass
class RoK4TerminationsCfg(TerminationsCfg):
    """Termination terms for RoK4 flat walking."""

    # Replace the parent's base-only contact term while keeping its time_out term.
    base_contact = None
    illegal_body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=ROK4_ILLEGAL_CONTACT_BODY_NAMES,
            ),
            "threshold": 1.0,
        },
    )


@configclass
class RoK4FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """RoK4 flat-ground velocity-tracking environment."""

    # Replace the parent locomotion reward set with RoK4-specific reward terms.
    rewards: RoK4RewardsCfg = RoK4RewardsCfg()
    # Keep the 13-dimensional interface while interpreting it in actuator coordinates.
    actions: RoK4ActionsCfg = RoK4ActionsCfg()
    observations: RoK4ObservationsCfg = RoK4ObservationsCfg()
    commands: RoK4CommandsCfg = RoK4CommandsCfg()
    # Keep the inherited timeout and replace base-only contact with RoK4 non-foot illegal contact.
    terminations: RoK4TerminationsCfg = RoK4TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # Start from Isaac Lab's common velocity-locomotion configuration, then override only RoK4-specific pieces.
        super().__post_init__()

        # Control timing. The policy sends a new action every decimation physics steps.
        # With dt=0.002 and decimation=5, physics runs at 500 Hz and policy/action updates at 100 Hz.
        self.decimation = 5
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            # Retain one contact-force sample per physics step across each policy interval.
            self.scene.contact_forces.class_type = RoK4ContactForceVisualizer
            self.scene.contact_forces.history_length = self.decimation
            self.scene.contact_forces.update_period = self.sim.dt
            # PhysX reports normal and tangential contact forces separately. Filter against the ground so the
            # visualizer can sum both components into one world-frame GRF vector for each foot.
            self.scene.contact_forces.filter_prim_paths_expr = [ROK4_GROUND_COLLISION_PRIM_PATH]
            self.scene.contact_forces.track_friction_forces = True

        # Scene. Use the RoK4 train asset and turn the inherited rough-terrain setup into a flat plane task.
        self.scene.robot = ROK4_TRAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # Actions and observations are actuator-space terms configured above. The single-frame observation remains
        # 48 values and history remains 5 frames, so actor/critic input dimensions stay unchanged at 240.

        # Events/randomization. The DR values live in domain_randomization_cfg.py for easier tuning.
        apply_rok4_domain_randomization(self)

        # Rewards. These weights adapt the inherited velocity-locomotion penalties to RoK4's body and joint names.
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["Base_Link", "Upper_Body_Link"]

        # Commands. Keep the initial flat-walking command range modest before moving to rough terrain.
        self.commands.base_velocity.ranges.lin_vel_x = ROK4_LIN_VEL_X_RANGE
        self.commands.base_velocity.ranges.lin_vel_y = ROK4_LIN_VEL_Y_RANGE
        self.commands.base_velocity.ranges.ang_vel_z = ROK4_ANG_VEL_Z_RANGE
        self.commands.base_velocity.rel_standing_envs = 0.05


@configclass
class RoK4FlatEnvCfg_PLAY(RoK4FlatEnvCfg):
    """Play configuration for RoK4 flat-ground velocity tracking."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.scene.robot = ROK4_TEST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.periodic_freeze_enabled = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class RoK4FlatEnvCfg_TELEOP(RoK4FlatEnvCfg_PLAY):
    """Teleoperation configuration for a trained RoK4 flat-ground policy."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.scene.num_envs = 1
        self.episode_length_s = 600.0

        # The teleoperation loop writes [vx, vy, wz] directly into the command term.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
