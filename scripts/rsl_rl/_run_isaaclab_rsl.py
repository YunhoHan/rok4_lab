"""Run Isaac Lab RSL-RL scripts with RoK4 tasks registered."""

from __future__ import annotations

import sys
from pathlib import Path


def _replace_once(source: str, marker: str, replacement: str, script_path: Path, purpose: str) -> str:
    """Replace one upstream source marker or raise a descriptive compatibility error."""
    if marker not in source:
        raise RuntimeError(f"Could not find {purpose} marker in {script_path}.")
    return source.replace(marker, replacement, 1)


def _prepare_isaaclab_rsl_source(
    script_path: Path, *, use_rok4_runner: bool = False, use_teleop: bool = False
) -> str:
    """Prepare an Isaac Lab RSL-RL script for local RoK4 execution."""
    source = script_path.read_text(encoding="utf-8")
    task_import = "import isaaclab_tasks  # noqa: F401\n"
    source = _replace_once(
        source,
        task_import,
        task_import + "import rok4_tasks  # noqa: F401\n",
        script_path,
        "Isaac Lab task import",
    )

    if use_rok4_runner:
        runner_import = "from rsl_rl.runners import DistillationRunner, OnPolicyRunner\n"
        source = _replace_once(
            source,
            runner_import,
            runner_import + "from rok4_ppo import RoK4OnPolicyRunner\n",
            script_path,
            "RSL-RL runner import",
        )
        source = _replace_once(
            source,
            "runner = OnPolicyRunner(",
            "runner = RoK4OnPolicyRunner(",
            script_path,
            "OnPolicyRunner construction",
        )

    if use_teleop:
        if script_path.name != "play.py":
            raise ValueError("RoK4 teleoperation injection is supported only for the RSL-RL play script.")

        parser_marker = "# append RSL-RL cli arguments\n"
        parser_args = '''parser.add_argument(
    "--teleop_device",
    type=str,
    choices=("gamepad", "keyboard"),
    default="gamepad",
    help="SE(2) command input device for RoK4 teleoperation.",
)
parser.add_argument(
    "--teleop_dead_zone",
    type=float,
    default=0.05,
    help="Normalized gamepad dead zone in the range [0, 1].",
)
'''
        source = _replace_once(
            source,
            parser_marker,
            parser_args + parser_marker,
            script_path,
            "RSL-RL CLI argument",
        )

        torch_import = "import torch\n"
        teleop_imports = '''from isaaclab.devices import Se2Gamepad, Se2GamepadCfg, Se2Keyboard, Se2KeyboardCfg

from rok4_tasks.manager_based.locomotion.velocity.config.rok4.flat_env_cfg import (
    ROK4_ANG_VEL_Z_RANGE,
    ROK4_LIN_VEL_X_RANGE,
    ROK4_LIN_VEL_Y_RANGE,
)
'''
        source = _replace_once(
            source,
            torch_import,
            torch_import + teleop_imports,
            script_path,
            "torch import",
        )

        main_marker = "@hydra_task_config(args_cli.task, args_cli.agent)\n"
        teleop_helpers = '''def _create_teleop_interface(device_name: str, sim_device: str, dead_zone: float):
    if not 0.0 <= dead_zone <= 1.0:
        raise ValueError(f"teleop_dead_zone must be in [0, 1], received {dead_zone}.")
    if device_name == "gamepad":
        cfg = Se2GamepadCfg(
            sim_device=sim_device,
            v_x_sensitivity=1.0,
            v_y_sensitivity=1.0,
            omega_z_sensitivity=1.0,
            dead_zone=dead_zone,
        )
        interface = Se2Gamepad(cfg)
    else:
        cfg = Se2KeyboardCfg(
            sim_device=sim_device,
            v_x_sensitivity=1.0,
            v_y_sensitivity=1.0,
            omega_z_sensitivity=1.0,
        )
        interface = Se2Keyboard(cfg)
    interface.reset()
    return interface


class _TeleopResetRequest:
    """Store a keyboard reset request until the inference loop can process it safely."""

    def __init__(self) -> None:
        self._requested = False

    def request(self) -> None:
        self._requested = True

    def consume(self) -> bool:
        requested = self._requested
        self._requested = False
        return requested


def _scale_teleop_command(raw_command: torch.Tensor, invert_lateral_and_yaw: bool = False) -> torch.Tensor:
    raw_command = torch.clamp(raw_command, -1.0, 1.0)
    command = torch.empty_like(raw_command)
    lateral_and_yaw_sign = -1.0 if invert_lateral_and_yaw else 1.0
    command[0] = torch.where(
        raw_command[0] >= 0.0,
        raw_command[0] * ROK4_LIN_VEL_X_RANGE[1],
        raw_command[0] * abs(ROK4_LIN_VEL_X_RANGE[0]),
    )
    command[1] = (
        lateral_and_yaw_sign * raw_command[1] * max(abs(value) for value in ROK4_LIN_VEL_Y_RANGE)
    )
    command[2] = (
        lateral_and_yaw_sign * raw_command[2] * max(abs(value) for value in ROK4_ANG_VEL_Z_RANGE)
    )
    return command


'''
        source = _replace_once(
            source,
            main_marker,
            teleop_helpers + main_marker,
            script_path,
            "Hydra main function",
        )

        wrapper_marker = "    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)\n"
        teleop_setup = '''
    teleop_interface = _create_teleop_interface(
        args_cli.teleop_device,
        sim_device=str(env.unwrapped.device),
        dead_zone=args_cli.teleop_dead_zone,
    )
    teleop_reset_request = _TeleopResetRequest()
    if args_cli.teleop_device == "keyboard":
        teleop_interface.add_callback("R", teleop_reset_request.request)
    base_velocity_command = env.unwrapped.command_manager.get_term("base_velocity")
    print(teleop_interface)
    if args_cli.teleop_device == "keyboard":
        print("\tReset environment: R")
'''
        source = _replace_once(
            source,
            wrapper_marker,
            wrapper_marker + teleop_setup,
            script_path,
            "RSL-RL environment wrapper",
        )

        inference_marker = "        # run everything in inference mode\n        with torch.inference_mode():\n"
        command_update = '''        if teleop_reset_request.consume():
            teleop_interface.reset()
            with torch.inference_mode():
                obs, _ = env.reset()
                reset_dones = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                policy_nn.reset(reset_dones)
            print("[INFO] Teleoperation environment reset.")
        raw_teleop_command = teleop_interface.advance()
        teleop_command = _scale_teleop_command(
            raw_teleop_command,
            invert_lateral_and_yaw=args_cli.teleop_device == "gamepad",
        )
        base_velocity_command.vel_command_b[:] = teleop_command
'''
        source = _replace_once(
            source,
            inference_marker,
            command_update + inference_marker,
            script_path,
            "inference loop",
        )

    return source


def run_isaaclab_rsl_script(
    script_name: str, use_rok4_runner: bool = False, use_teleop: bool = False
) -> None:
    """Execute an Isaac Lab RSL-RL script after registering RoK4 tasks."""
    rok4_lab_dir = Path(__file__).resolve().parents[2]
    rok4_source_dir = rok4_lab_dir / "source" / "rok4_tasks"
    if str(rok4_source_dir) not in sys.path:
        sys.path.insert(0, str(rok4_source_dir))

    isaaclab_dir = Path.cwd()
    script_path = isaaclab_dir / "scripts" / "reinforcement_learning" / "rsl_rl" / script_name
    rsl_script_dir = script_path.parent
    if str(rsl_script_dir) not in sys.path:
        sys.path.insert(0, str(rsl_script_dir))
    if not script_path.exists():
        raise FileNotFoundError(
            f"Could not find Isaac Lab RSL-RL script at {script_path}. "
            "Run this wrapper from the root of your Isaac Lab checkout."
        )

    source = _prepare_isaaclab_rsl_source(
        script_path,
        use_rok4_runner=use_rok4_runner,
        use_teleop=use_teleop,
    )

    globals_dict = {
        "__file__": str(script_path),
        "__name__": "__main__",
        "__package__": None,
    }
    exec(compile(source, str(script_path), "exec"), globals_dict)
