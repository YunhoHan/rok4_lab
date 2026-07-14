"""RSL-RL PPO configurations for RoK4 velocity tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RoK4FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the RoK4 flat velocity task."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "rok4_flat"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    # Match the Isaac Gym/RL-Games setup: clip policy actions before the environment applies action scale.
    clip_actions = 1.0
    # Initial RoK4 baseline: RoK4-oriented network/normalization settings with G1-style PPO algorithm values.
    # These values are starting points for flat walking, not final tuned parameters.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
