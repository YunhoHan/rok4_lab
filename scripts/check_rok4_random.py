"""Spawn RoK4 and apply small sinusoidal position commands."""

import argparse
import math
import sys
from pathlib import Path

ROK4_TASKS_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "rok4_tasks"
if str(ROK4_TASKS_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(ROK4_TASKS_SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check RoK4 with small sinusoidal joint commands.")
parser.add_argument("--asset", choices=["train", "test"], default="train", help="RoK4 USD variant to spawn.")
parser.add_argument("--amplitude", type=float, default=0.05, help="Joint target amplitude [rad].")
parser.add_argument("--frequency", type=float, default=0.5, help="Joint command frequency [Hz].")
parser.add_argument("--fix_root", action="store_true", help="Fix the root link when spawning the articulation.")
parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity for all rigid bodies.")
parser.add_argument("--hold_root", action="store_true", help="Rewrite the root pose and velocity every step.")
parser.add_argument("--root_height", type=float, default=None, help="Override the initial root height [m].")
parser.add_argument("--reset_interval", type=int, default=1500, help="Simulation steps between state resets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from rok4_tasks.assets.robots import ROK4_TEST_CFG, ROK4_TRAIN_CFG


def design_scene() -> tuple[dict[str, Articulation], torch.Tensor]:
    """Create the ground, lighting, and one RoK4 articulation."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    origin = torch.tensor([[0.0, 0.0, 0.0]], device=args_cli.device)
    sim_utils.create_prim("/World/Origin", "Xform", translation=origin[0].tolist())

    robot_cfg = (ROK4_TEST_CFG if args_cli.asset == "test" else ROK4_TRAIN_CFG).copy()
    robot_cfg.prim_path = "/World/Origin/Robot"
    if args_cli.root_height is not None:
        robot_cfg.init_state.pos = (robot_cfg.init_state.pos[0], robot_cfg.init_state.pos[1], args_cli.root_height)
    if args_cli.disable_gravity:
        robot_cfg.spawn.rigid_props.disable_gravity = True
    if args_cli.fix_root:
        robot_cfg.spawn.articulation_props.fix_root_link = True
    robot = Articulation(cfg=robot_cfg)

    return {"robot": robot}, origin


def reset_robot(robot: Articulation, origin: torch.Tensor):
    """Reset RoK4 to the configured initial pose."""
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origin: torch.Tensor):
    """Run the simulation loop."""
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % args_cli.reset_interval == 0:
            count = 0
            reset_robot(robot, origin)
            print(f"[INFO]: Resetting RoK4 state. asset={args_cli.asset}, amplitude={args_cli.amplitude}")

        phase = 2.0 * math.pi * args_cli.frequency * count * sim_dt
        joint_offsets = torch.arange(robot.num_joints, device=sim.device, dtype=torch.float32).unsqueeze(0) * 0.45
        joint_target = robot.data.default_joint_pos + args_cli.amplitude * torch.sin(phase + joint_offsets)

        # IdealPDActuatorCfg converts these targets to joint efforts:
        # tau = Kp * (q_des - q) + Kd * (qd_des - qd).
        robot.set_joint_position_target(joint_target)
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        if args_cli.hold_root:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origin
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        robot.write_data_to_sim()

        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Run the RoK4 sinusoidal-command check."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, -2.2, 1.6], [0.0, 0.0, 0.55])

    scene_entities, origin = design_scene()
    sim.reset()
    print("[INFO]: RoK4 sinusoidal-command check setup complete.")
    run_simulator(sim, scene_entities, origin)


if __name__ == "__main__":
    main()
    simulation_app.close()
