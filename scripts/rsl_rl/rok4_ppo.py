# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4 RSL-RL extensions that add PPO KL-divergence logging."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner


class RoK4PPO(PPO):
    """RSL-RL PPO with per-iteration KL statistics in the loss dictionary."""

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

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
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
    """Construct the standard RSL-RL policy with :class:`RoK4PPO`."""

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
        }
        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name not in policy_classes:
            raise ValueError(f"Unsupported RoK4 policy class: {policy_class_name}.")
        actor_critic = policy_classes[policy_class_name](
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        algorithm_class_name = self.alg_cfg.pop("class_name")
        if algorithm_class_name != "PPO":
            raise ValueError(f"RoK4OnPolicyRunner requires PPO, received {algorithm_class_name}.")
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
