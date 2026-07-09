"""Gym registrations for RoK4 velocity tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RoK4-Isaac-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RoK4FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoK4FlatPPORunnerCfg",
    },
)


gym.register(
    id="RoK4-Isaac-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:RoK4FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoK4FlatPPORunnerCfg",
    },
)
