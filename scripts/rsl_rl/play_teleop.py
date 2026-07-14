"""Teleoperate a trained RoK4 RSL-RL policy with an SE(2) input device."""

from _run_isaaclab_rsl import run_isaaclab_rsl_script


run_isaaclab_rsl_script("play.py", use_teleop=True)
