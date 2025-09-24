from __future__ import annotations

import torch
import torch.nn.functional as F

from agents.ppo.ppo_agent import PPOAgent
from utils.replay_buffer import ReplayBuffer
from utils.torch import assert_detached

# TODO: cleanup, learn how to use this

class P3OAgent(PPOAgent):
    """P3O-style agent built on PPO with optional replay.

    Moves replay/off-policy mixing out of PPOAgent and into this subclass.
    Keeps PPO’s GAE, value, and entropy settings intact.
    """

    def __init__(self, config):
        super().__init__(config)
        # Optional replay buffer for off-policy updates
        self._replay_buffer = ReplayBuffer(capacity=int(self.config.replay_buffer_size))

    def on_train_epoch_start(self):
        # Collect fresh rollout (handled by super), then push to replay if present
        super().on_train_epoch_start()
        self._replay_buffer.add_trajectories(self._trajectories)

    def training_step(self, batch, batch_idx):
        # Standard on-policy PPO update
        if getattr(self, "_early_stop_epoch", False):
            return None

        result = self.losses_for_batch(batch, batch_idx)
        if result["early_stop_epoch"]:
            self._early_stop_epoch = True
            return None

        self._backpropagate_and_step(result["loss"])

        # Additional off-policy updates from replay (clipped IS)
        n_extra = int( self.config.replay_ratio)
        batch_size = self.config.batch_size
        for _ in range(max(0, n_extra)):
            off_b = self._replay_buffer.sample(batch_size, device=batch.observations.device)
            loss_off = self.losses_for_offpolicy_batch(off_b, batch_idx)
            self._backpropagate_and_step(loss_off)

        return None
    
    # TODO: there is a lot of overlap with PPO loss calc, encapsulate and reuse
    def losses_for_offpolicy_batch(self, batch, batch_idx):
        # PPO loss with importance-sampling clipping for off-policy samples
        states = batch.observations
        actions = batch.actions
        old_logprobs = batch.logprobs
        advantages = batch.advantages
        returns = batch.returns

        assert_detached(states, actions, old_logprobs, advantages, returns)

        normalize_advantages = self.config.normalize_advantages == "batch"
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_dist, values_pred = self.policy_model(states)
        new_logprobs = policy_dist.log_prob(actions)

        ratio = torch.exp(new_logprobs - old_logprobs)
        is_cap = float(getattr(self.config, "replay_is_clip", 10.0) or 10.0)
        if is_cap is not None and is_cap > 0:
            ratio = torch.clamp(ratio, max=is_cap)

        scaled_advantages = advantages * ratio
        scaled_advantages_clamped = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        min_scaled_advantages = torch.min(scaled_advantages, scaled_advantages_clamped)
        policy_loss = -min_scaled_advantages.mean()

        value_loss = F.mse_loss(values_pred, returns)
        entropy = policy_dist.entropy().mean()
        entropy_loss = -entropy

        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        with torch.no_grad():
            self.metrics_recorder.record("train", {
                'opt_replay/loss/total': loss.detach(),
                'opt_replay/loss/policy': policy_loss.detach(),
                'opt_replay/loss/value': value_loss.detach(),
                'opt_replay/loss/entropy': entropy_loss.detach(),
            })

        return loss

