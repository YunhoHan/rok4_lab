"""RSL-RL PPO configurations for RoK4 velocity tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from rok4_tasks.manager_based.locomotion.velocity.mdp.symmetry import compute_symmetric_states


@configclass
class RoK4EstimatorActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic configuration with a command-free base-velocity estimator."""

    class_name: str = "RoK4EstimatorActorCritic"
    estimator_hidden_dims: list[int] = [256, 128]
    estimator_activation: str = "elu"
    estimator_command_history_start: int = 30
    estimator_command_history_end: int = 45


@configclass
class RoK4EstimatorPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with separately supervised estimator optimization."""

    class_name: str = "RoK4PPO"
    estimator_learning_rate: float = 1.0e-3
    estimator_max_grad_norm: float = 1.0


@configclass
class RoK4FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the RoK4 flat velocity task."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "rok4_flat"
    obs_groups = {"policy": ["policy"], "critic": ["critic", "privileged"]}
    # Match the Isaac Gym/RL-Games setup: clip policy actions before the environment applies action scale.
    clip_actions = 1.0
    # Initial RoK4 baseline: RoK4-oriented network/normalization settings with G1-style PPO algorithm values.
    # These values are starting points for flat walking, not final tuned parameters.
    policy = RoK4EstimatorActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RoK4EstimatorPpoAlgorithmCfg(
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff=0.0,
        ),
    )
