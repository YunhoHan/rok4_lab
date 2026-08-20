"""Play RoK4 RSL-RL tasks through Isaac Lab's play script."""

from _run_isaaclab_rsl import run_isaaclab_rsl_script


run_isaaclab_rsl_script("play.py", use_rok4_runner=True, use_push_ui=True)
