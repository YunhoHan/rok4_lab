"""Run Isaac Lab RSL-RL scripts with RoK4 tasks registered."""

from __future__ import annotations

import sys
from pathlib import Path


def run_isaaclab_rsl_script(script_name: str) -> None:
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

    source = script_path.read_text(encoding="utf-8")
    marker = "import isaaclab_tasks  # noqa: F401\n"
    if marker not in source:
        raise RuntimeError(f"Could not find Isaac Lab task import marker in {script_path}.")
    source = source.replace(marker, marker + "import rok4_tasks  # noqa: F401\n", 1)

    globals_dict = {
        "__file__": str(script_path),
        "__name__": "__main__",
        "__package__": None,
    }
    exec(compile(source, str(script_path), "exec"), globals_dict)
