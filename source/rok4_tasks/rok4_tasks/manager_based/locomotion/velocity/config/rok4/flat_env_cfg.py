"""Flat velocity-tracking task for RoK4."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from rok4_tasks.assets.robots.rok4 import ROK4_ACTION_SCALE, ROK4_JOINT_ORDER, ROK4_TEST_CFG, ROK4_TRAIN_CFG


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
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch_Joint", ".*_Ankle_Roll_Joint"])},
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


@configclass
class RoK4FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """RoK4 flat-ground velocity-tracking environment."""

    # Replace the parent locomotion reward set with RoK4-specific reward terms.
    rewards: RoK4RewardsCfg = RoK4RewardsCfg()

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
            self.scene.contact_forces.update_period = self.sim.dt

        # Scene. Use the RoK4 train asset and turn the inherited rough-terrain setup into a flat plane task.
        self.scene.robot = ROK4_TRAIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # Actions. The policy output is interpreted as joint-position offsets around the default pose.
        self.actions.joint_pos.joint_names = ROK4_JOINT_ORDER
        self.actions.joint_pos.scale = ROK4_ACTION_SCALE
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.use_default_offset = True

        # Blind policy observations. Keep base linear velocity out of the actor input.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = True
        self.observations.policy.enable_corruption = True
        self.observations.policy.concatenate_terms = True

        # Events/randomization. Apply inherited mass, COM, external-force, joint-reset, and base-reset events to RoK4
        # body names and use mild reset ranges for the first flat walking baseline.
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Base_Link"]
        self.events.base_com.params["asset_cfg"].body_names = ["Base_Link"]
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Base_Link"]
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            },
        }

        # Rewards. These weights adapt the inherited velocity-locomotion penalties to RoK4's body and joint names.
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch_Joint"]
        )
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch_Joint", ".*_Ankle_.*"]
        )
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["Base_Link", "Upper_Body_Link"]

        # Commands. Keep the initial flat-walking command range modest before moving to rough terrain.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # Terminations. Treat base or upper-body contact as falling.
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Base_Link", "Upper_Body_Link"]


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
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
