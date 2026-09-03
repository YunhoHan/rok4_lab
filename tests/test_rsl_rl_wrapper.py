# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RoK4's local Isaac Lab RSL-RL source wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_WRAPPER_PATH = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/_run_isaaclab_rsl.py"
_SPEC = importlib.util.spec_from_file_location("rok4_rsl_wrapper_under_test", _WRAPPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WRAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WRAPPER)


def _upstream_script(tmp_path: Path, name: str) -> Path:
    source = (
        "from rsl_rl.runners import DistillationRunner, OnPolicyRunner\n"
        "import isaaclab_tasks  # noqa: F401\n"
        "runner = OnPolicyRunner(env, cfg)\n"
    )
    if name == "play.py":
        source += (
            "export_policy_as_jit(policy_nn, normalizer=normalizer, "
            'path=export_model_dir, filename="policy.pt")\n'
            "export_policy_as_onnx(policy_nn, normalizer=normalizer, "
            'path=export_model_dir, filename="policy.onnx")\n'
        )
    script_path = tmp_path / name
    script_path.write_text(source, encoding="utf-8")
    return script_path


def _upstream_teleop_script(tmp_path: Path) -> Path:
    source = (
        "from rsl_rl.runners import DistillationRunner, OnPolicyRunner\n"
        "import isaaclab_tasks  # noqa: F401\n"
        "# append RSL-RL cli arguments\n"
        "import torch\n"
        "@hydra_task_config(args_cli.task, args_cli.agent)\n"
        "def main(env_cfg, agent_cfg):\n"
        "    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)\n"
        "    runner = OnPolicyRunner(env, cfg)\n"
        "    policy_nn = runner.alg.policy\n"
        "    export_policy_as_jit(policy_nn, normalizer=normalizer, "
        'path=export_model_dir, filename="policy.pt")\n'
        "    export_policy_as_onnx(policy_nn, normalizer=normalizer, "
        'path=export_model_dir, filename="policy.onnx")\n'
        "    while True:\n"
        "        # run everything in inference mode\n"
        "        with torch.inference_mode():\n"
        "                    # agent stepping\n"
        "                    actions = policy(obs)\n"
        "                    obs, _, dones, _ = env.step(actions)\n"
    )
    script_path = tmp_path / "play.py"
    script_path.write_text(source, encoding="utf-8")
    return script_path


def test_train_source_uses_rok4_estimator_runner(tmp_path: Path) -> None:
    """Training must construct the runner that owns the estimator optimizer."""
    source = _WRAPPER._prepare_isaaclab_rsl_source(
        _upstream_script(tmp_path, "train.py"), use_rok4_runner=True
    )

    assert "runner = RoK4OnPolicyRunner(" in source
    compile(source, "train.py", "exec")


def test_play_source_uses_rok4_runner_and_fused_exporters(tmp_path: Path) -> None:
    """Play must load estimator checkpoints and export a fused 240D policy."""
    source = _WRAPPER._prepare_isaaclab_rsl_source(
        _upstream_script(tmp_path, "play.py"), use_rok4_runner=True
    )

    assert "runner = RoK4OnPolicyRunner(" in source
    assert "export_rok4_policy_as_jit(" in source
    assert "export_rok4_policy_as_onnx(" in source
    compile(source, "play.py", "exec")


def test_teleop_source_visualizes_actor_velocity_estimate(tmp_path: Path) -> None:
    """Teleop must draw the estimator output computed from the current policy observation."""
    source = _WRAPPER._prepare_isaaclab_rsl_source(
        _upstream_teleop_script(tmp_path),
        use_rok4_runner=True,
        use_teleop=True,
    )

    assert "RoK4EstimatedVelocityVisualizer" in source
    assert "estimated_base_lin_vel_b = policy_nn.estimate_base_velocity(obs)" in source
    assert "estimated_velocity_visualizer.visualize(estimated_base_lin_vel_b)" in source
    assert "obs, _, dones, infos = env.step(actions)" in source
    assert "_print_touchdown_diagnostics(infos)" in source
    compile(source, "play.py", "exec")
