# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4 RSL-RL extensions for KL logging and concurrent velocity estimation."""

from __future__ import annotations

import copy
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.networks import MLP
from rsl_rl.runners import OnPolicyRunner


ROK4_POLICY_OBS_DIM = 240
ROK4_COMMAND_HISTORY_START = 30
ROK4_COMMAND_HISTORY_END = 45
ROK4_ESTIMATOR_OBS_DIM = 225
ROK4_ESTIMATOR_OUTPUT_DIM = 3
ROK4_PRIVILEGED_OBS_DIM = 10


class RoK4VelocityEstimatorActor(nn.Module):
    """Estimate base velocity and feed it to the actuator-space policy.

    The public input remains the 240-value actor observation. The estimator
    removes the 15-value command history and predicts body-frame base linear
    velocity from the remaining 225 proprioceptive values. Its estimate is
    detached before entering the policy MLP so PPO cannot train the estimator.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple[int, ...],
        estimator_hidden_dims: list[int] | tuple[int, ...],
        activation: str,
        estimator_activation: str,
        command_history_start: int,
        command_history_end: int,
    ) -> None:
        super().__init__()
        if num_actor_obs != ROK4_POLICY_OBS_DIM:
            raise ValueError(f"Expected {ROK4_POLICY_OBS_DIM} actor observations, received {num_actor_obs}.")
        if not 0 <= command_history_start < command_history_end <= num_actor_obs:
            raise ValueError(
                "Estimator command-history slice must lie inside the actor observation, received "
                f"[{command_history_start}:{command_history_end}] for {num_actor_obs} values."
            )

        estimator_obs_dim = num_actor_obs - (command_history_end - command_history_start)
        if estimator_obs_dim != ROK4_ESTIMATOR_OBS_DIM:
            raise ValueError(
                f"Expected {ROK4_ESTIMATOR_OBS_DIM} estimator observations, received {estimator_obs_dim}."
            )

        self.input_dim = num_actor_obs
        self.command_history_start = command_history_start
        self.command_history_end = command_history_end
        self.estimator = MLP(
            estimator_obs_dim,
            ROK4_ESTIMATOR_OUTPUT_DIM,
            estimator_hidden_dims,
            estimator_activation,
        )
        self.policy = MLP(
            num_actor_obs + ROK4_ESTIMATOR_OUTPUT_DIM,
            num_actions,
            actor_hidden_dims,
            activation,
        )

    def estimator_observation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Remove the command history from normalized actor observations."""
        return torch.cat(
            (
                actor_obs[..., : self.command_history_start],
                actor_obs[..., self.command_history_end :],
            ),
            dim=-1,
        )

    def estimate(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Estimate body-frame base linear velocity [m/s]."""
        return self.estimator(self.estimator_observation(actor_obs))

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Return actuator actions while isolating estimator gradients from PPO."""
        actions, _ = self.action_and_estimate(actor_obs)
        return actions

    def action_and_estimate(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return actions and the detached velocity estimate used by the policy."""
        estimated_base_velocity = self.estimate(actor_obs).detach()
        actions = self.policy(torch.cat((actor_obs, estimated_base_velocity), dim=-1))
        return actions, estimated_base_velocity


class RoK4EstimatorActorCritic(ActorCritic):
    """Feed-forward actor-critic with a supervised body-velocity estimator."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        estimator_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        estimator_activation: str = "elu",
        estimator_command_history_start: int = ROK4_COMMAND_HISTORY_START,
        estimator_command_history_end: int = ROK4_COMMAND_HISTORY_END,
        **kwargs,
    ) -> None:
        if kwargs.get("state_dependent_std", False):
            raise ValueError("RoK4EstimatorActorCritic does not support state-dependent action standard deviation.")

        actor_hidden_dims = kwargs.get("actor_hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        if num_critic_obs != ROK4_POLICY_OBS_DIM + ROK4_PRIVILEGED_OBS_DIM:
            raise ValueError(
                f"Expected {ROK4_POLICY_OBS_DIM + ROK4_PRIVILEGED_OBS_DIM} critic values, "
                f"received {num_critic_obs}."
            )
        if "privileged" not in obs.keys() or obs["privileged"].shape[-1] != ROK4_PRIVILEGED_OBS_DIM:
            raise ValueError(
                f"RoK4 estimator training requires a {ROK4_PRIVILEGED_OBS_DIM}-value privileged observation group."
            )

        self.actor = RoK4VelocityEstimatorActor(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            estimator_hidden_dims=estimator_hidden_dims,
            activation=activation,
            estimator_activation=estimator_activation,
            command_history_start=estimator_command_history_start,
            command_history_end=estimator_command_history_end,
        )
        print(f"RoK4 estimator MLP: {self.actor.estimator}")
        print(f"RoK4 actor MLP: {self.actor.policy}")

    def estimate_base_velocity(self, obs: TensorDict) -> torch.Tensor:
        """Estimate body-frame base linear velocity from noisy policy history [m/s]."""
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor.estimate(actor_obs)

    @staticmethod
    def estimator_target(obs: TensorDict) -> torch.Tensor:
        """Return uncorrupted simulator base linear velocity targets [m/s]."""
        if "privileged" not in obs.keys() or obs["privileged"].shape[-1] < ROK4_ESTIMATOR_OUTPUT_DIM:
            raise ValueError("Estimator targets require privileged base linear velocity in indices [0:3].")
        return obs["privileged"][..., :ROK4_ESTIMATOR_OUTPUT_DIM]

    def estimator_parameters(self) -> list[nn.Parameter]:
        """Return parameters optimized only by supervised estimator learning."""
        return list(self.actor.estimator.parameters())

    def ppo_parameters(self) -> list[nn.Parameter]:
        """Return policy parameters excluding the supervised estimator."""
        estimator_parameter_ids = {id(parameter) for parameter in self.estimator_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in estimator_parameter_ids]


class RoK4PPO(PPO):
    """RSL-RL PPO with KL logging and separately supervised velocity estimation."""

    def __init__(
        self,
        policy: RoK4EstimatorActorCritic,
        estimator_learning_rate: float = 1.0e-3,
        estimator_max_grad_norm: float = 1.0,
        **kwargs,
    ) -> None:
        if not isinstance(policy, RoK4EstimatorActorCritic):
            raise TypeError(f"RoK4PPO requires RoK4EstimatorActorCritic, received {type(policy).__name__}.")
        super().__init__(policy, **kwargs)
        self.optimizer = optim.Adam(policy.ppo_parameters(), lr=self.learning_rate)
        self.estimator_optimizer = optim.Adam(policy.estimator_parameters(), lr=estimator_learning_rate)
        self.estimator_learning_rate = estimator_learning_rate
        self.estimator_max_grad_norm = estimator_max_grad_norm
        self.estimator_metrics: dict[str, float] = {}

    def update(self) -> dict[str, float]:
        """Update the policy and return losses plus adaptive-schedule KL statistics."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        max_kl = 0.0
        num_kl_updates = 0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_estimator_loss = 0.0
        mean_estimator_squared_error = torch.zeros(ROK4_ESTIMATOR_OUTPUT_DIM, device=self.device)

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    kl_value = kl_mean.item()
                    mean_kl += kl_value
                    max_kl = max(max_kl, kl_value)
                    num_kl_updates += 1

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.learning_rate = learning_rate.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mse_loss = torch.nn.MSELoss()
                rnd_loss = mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            self.estimator_optimizer.zero_grad(set_to_none=True)
            estimated_base_velocity = self.policy.estimate_base_velocity(obs_batch)
            estimator_target = self.policy.estimator_target(obs_batch)
            estimator_error = estimated_base_velocity - estimator_target
            estimator_loss = F.mse_loss(estimated_base_velocity, estimator_target)
            estimator_loss.backward()
            if self.is_multi_gpu:
                estimator_gradients = [
                    parameter.grad.reshape(-1)
                    for parameter in self.policy.estimator_parameters()
                    if parameter.grad is not None
                ]
                flattened_gradients = torch.cat(estimator_gradients)
                torch.distributed.all_reduce(flattened_gradients, op=torch.distributed.ReduceOp.SUM)
                flattened_gradients /= self.gpu_world_size
                offset = 0
                for parameter in self.policy.estimator_parameters():
                    if parameter.grad is not None:
                        numel = parameter.numel()
                        parameter.grad.copy_(flattened_gradients[offset : offset + numel].view_as(parameter.grad))
                        offset += numel
            nn.utils.clip_grad_norm_(self.policy.estimator_parameters(), self.estimator_max_grad_norm)
            self.estimator_optimizer.step()
            self.estimator_optimizer.zero_grad(set_to_none=True)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            mean_estimator_loss += estimator_loss.item()
            mean_estimator_squared_error += torch.mean(torch.square(estimator_error.detach()), dim=0)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_estimator_squared_error /= num_updates
        estimator_rmse = torch.sqrt(mean_estimator_squared_error)
        self.estimator_metrics = {
            "rmse_vx": estimator_rmse[0].item(),
            "rmse_vy": estimator_rmse[1].item(),
            "rmse_vz": estimator_rmse[2].item(),
            "rmse_total": torch.sqrt(torch.sum(mean_estimator_squared_error)).item(),
        }

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimator_mse": mean_estimator_loss,
        }
        if num_kl_updates > 0:
            loss_dict["kl"] = mean_kl / num_kl_updates
            loss_dict["kl_max"] = max_kl
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


class RoK4OnPolicyRunner(OnPolicyRunner):
    """Construct, log, save, and load the RoK4 estimator policy."""

    def _construct_algorithm(self, obs: TensorDict) -> RoK4PPO:
        """Construct the actor-critic policy and the KL-logging PPO algorithm."""
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Set actor and critic observation "
                "normalization in the policy configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        policy_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
            "RoK4EstimatorActorCritic": RoK4EstimatorActorCritic,
        }
        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name not in policy_classes:
            raise ValueError(f"Unsupported RoK4 policy class: {policy_class_name}.")
        actor_critic = policy_classes[policy_class_name](
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        if algorithm_class_name not in {"PPO", "RoK4PPO"}:
            raise ValueError(f"RoK4OnPolicyRunner requires RoK4PPO, received {algorithm_class_name}.")
        algorithm = RoK4PPO(
            actor_critic,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Log standard runner values and physical-unit estimator RMSE."""
        super().log(locs, width=width, pad=pad)
        for key, value in self.alg.estimator_metrics.items():
            self.writer.add_scalar(f"Metrics/estimator/{key}", value, locs["it"])

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save policy, PPO optimizer, and estimator optimizer states."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "estimator_optimizer_state_dict": self.alg.estimator_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        """Load policy and both optimizer states from a RoK4 checkpoint."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]


class _RoK4PolicyExporter(nn.Module):
    """Export the normalized 240D estimator-policy composition as one module."""

    def __init__(self, policy: RoK4EstimatorActorCritic, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        if not isinstance(policy, RoK4EstimatorActorCritic):
            raise TypeError(f"Expected RoK4EstimatorActorCritic, received {type(policy).__name__}.")
        self.actor = copy.deepcopy(policy.actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Map one 240-value actor observation to 13 actuator actions."""
        return self.actor(self.normalizer(observation))

    @torch.jit.export
    def reset(self) -> None:
        """Retain the standard stateless exported-policy interface."""
        pass


class _RoK4OnnxPolicyExporter(_RoK4PolicyExporter):
    """Expose the action and its internal base-velocity estimate as ONNX outputs."""

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map one 240-value observation to actions and estimated velocity [unitless, m/s]."""
        normalized_observation = self.normalizer(observation)
        return self.actor.action_and_estimate(normalized_observation)


def export_rok4_policy_as_jit(
    policy: RoK4EstimatorActorCritic,
    normalizer: nn.Module | None,
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Export a fused 240D RoK4 estimator policy as TorchScript."""
    os.makedirs(path, exist_ok=True)
    exporter = _RoK4PolicyExporter(policy, normalizer).cpu().eval()
    torch.jit.script(exporter).save(os.path.join(path, filename))


def export_rok4_policy_as_onnx(
    policy: RoK4EstimatorActorCritic,
    path: str,
    normalizer: nn.Module | None = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export a fused policy with separate action and estimated-velocity outputs."""
    os.makedirs(path, exist_ok=True)
    exporter = _RoK4OnnxPolicyExporter(policy, normalizer).cpu().eval()
    observation = torch.zeros(1, ROK4_POLICY_OBS_DIM)
    torch.onnx.export(
        exporter,
        observation,
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions", "estimated_base_lin_vel_b"],
        dynamic_axes={},
    )
