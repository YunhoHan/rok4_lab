"""Move RoK4 joints one at a time for visual and torque-PD checks."""

import argparse
import math
import sys
from pathlib import Path

ROK4_TASKS_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "rok4_tasks"
if str(ROK4_TASKS_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(ROK4_TASKS_SOURCE_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check RoK4 joints one at a time.")
parser.add_argument("--asset", choices=["train", "test"], default="test", help="RoK4 USD variant to spawn.")
parser.add_argument("--joint", type=str, default="all", help="Exact joint name to move, or 'all' to sweep all joints.")
parser.add_argument(
    "--mode",
    choices=["teleport", "torque_pd"],
    default="torque_pd",
    help="Teleport joint state directly or command it through explicit torque PD.",
)
parser.add_argument(
    "--motion",
    choices=["limits", "amplitude"],
    default="limits",
    help="Move through exact joint limits or use a custom amplitude around a center.",
)
parser.add_argument("--center", type=float, default=None, help="Joint motion center [rad] for amplitude motion.")
parser.add_argument("--amplitude", type=float, default=0.25, help="Joint motion amplitude [rad] for amplitude motion.")
parser.add_argument("--frequency", type=float, default=0.2, help="Joint command frequency [Hz] for amplitude motion.")
parser.add_argument("--joint_duration", type=float, default=5.0, help="Time spent moving each joint [s].")
parser.add_argument("--fix_root", action="store_true", help="Fix the root link when spawning the articulation.")
parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity for all rigid bodies.")
parser.add_argument("--hold_root", action="store_true", help="Rewrite the root pose and velocity every step.")
parser.add_argument("--root_height", type=float, default=1.2, help="Override the initial root height [m].")
parser.add_argument("--reset_interval", type=int, default=0, help="Simulation steps between state resets. Use 0 to disable.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from rok4_tasks.assets.robots import ROK4_TEST_CFG, ROK4_TRAIN_CFG
from rok4_tasks.assets.robots.rok4 import ROK4_JOINT_ORDER


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


def hold_root_pose(robot: Articulation, origin: torch.Tensor):
    """Keep the root pose fixed by rewriting root state."""
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origin
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))


def resolve_joint_sequence(robot: Articulation) -> tuple[list[int], list[str]]:
    """Resolve the joint sequence to move."""
    if args_cli.joint.lower() == "all":
        joint_ids, joint_names = robot.find_joints(ROK4_JOINT_ORDER, preserve_order=True)
        missing_joint_names = [joint_name for joint_name in ROK4_JOINT_ORDER if joint_name not in joint_names]
        if missing_joint_names:
            print(f"[WARN]: Missing joints skipped: {missing_joint_names}", flush=True)
        return joint_ids, joint_names

    joint_ids, joint_names = robot.find_joints([args_cli.joint], preserve_order=True)
    if len(joint_ids) != 1:
        raise RuntimeError(f"Expected exactly one joint for '{args_cli.joint}', found: {joint_names}")
    return joint_ids, joint_names


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origin: torch.Tensor):
    """Run the joint monkey loop."""
    robot = entities["robot"]
    sim_dt = sim.get_physics_dt()
    steps_per_joint = max(1, int(round(args_cli.joint_duration / sim_dt)))
    count = 0
    last_joint_index = None

    joint_ids, joint_names = resolve_joint_sequence(robot)
    if not joint_ids:
        raise RuntimeError("No joints were selected for the RoK4 joint monkey check.")

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    joint_pos_limits = robot.data.joint_pos_limits[0, joint_ids].clone()
    centers = []
    amplitudes = []
    for sequence_index, joint_id in enumerate(joint_ids):
        if args_cli.motion == "limits":
            lower = joint_pos_limits[sequence_index, 0].item()
            upper = joint_pos_limits[sequence_index, 1].item()
            centers.append(0.5 * (lower + upper))
            amplitudes.append(0.5 * (upper - lower))
        else:
            centers.append(args_cli.center if args_cli.center is not None else default_joint_pos[0, joint_id].item())
            amplitudes.append(args_cli.amplitude)

    print(
        "[INFO]: Joint monkey setup complete. "
        f"joints={len(joint_ids)}, mode={args_cli.mode}, motion={args_cli.motion}, "
        f"joint_duration={args_cli.joint_duration:.2f}",
        flush=True,
    )
    print(f"[INFO]: Joint sequence: {joint_names}", flush=True)

    while simulation_app.is_running():
        if count == 0 or (args_cli.reset_interval > 0 and count % args_cli.reset_interval == 0):
            reset_robot(robot, origin)
            print(f"[INFO]: Resetting RoK4 state. mode={args_cli.mode}", flush=True)

        joint_index = (count // steps_per_joint) % len(joint_ids)
        step_in_joint = count % steps_per_joint
        joint_id = joint_ids[joint_index]
        joint_name = joint_names[joint_index]
        center = centers[joint_index]
        amplitude = amplitudes[joint_index]
        if joint_index != last_joint_index:
            last_joint_index = joint_index
            if args_cli.motion == "limits":
                lower = joint_pos_limits[joint_index, 0].item()
                upper = joint_pos_limits[joint_index, 1].item()
                print(
                    f"[INFO]: Moving joint {joint_index + 1}/{len(joint_ids)}: "
                    f"{joint_name} [{lower:.4f}, {upper:.4f}] rad",
                    flush=True,
                )
            else:
                print(
                    f"[INFO]: Moving joint {joint_index + 1}/{len(joint_ids)}: "
                    f"{joint_name} center={center:.4f}, amplitude={amplitude:.4f} rad",
                    flush=True,
                )

        if args_cli.motion == "limits":
            phase = 2.0 * math.pi * step_in_joint / steps_per_joint
        else:
            phase = 2.0 * math.pi * args_cli.frequency * step_in_joint * sim_dt
        target = center + amplitude * math.sin(phase)

        if args_cli.mode == "teleport":
            joint_pos = default_joint_pos.clone()
            joint_vel = default_joint_vel.clone()
            joint_pos[:, joint_id] = target
            joint_vel[:, joint_id] = 0.0
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        else:
            joint_target = default_joint_pos.clone()
            joint_target[:, joint_id] = target
            robot.set_joint_position_target(joint_target)
            robot.set_joint_velocity_target(torch.zeros_like(default_joint_vel))
            robot.write_data_to_sim()

        if args_cli.hold_root:
            hold_root_pose(robot, origin)

        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Run the RoK4 joint monkey check."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.2, -2.2, 1.6], [0.0, 0.0, 0.65])

    scene_entities, origin = design_scene()
    sim.reset()
    run_simulator(sim, scene_entities, origin)


if __name__ == "__main__":
    main()
    simulation_app.close()
