import torch
import torch.nn.functional as F

from utils.torch import assert_detached, batch_normalize, compute_kl_diagnostics, compute_kl_metrics, normalize_batch_with_metrics

from ..base_agent import BaseAgent
from .ppo_alerts import PPOAlerts


class PPOAgent(BaseAgent):
    
    def __init__(self, config):
        super().__init__(config)

        # Initialize PPO-specific hyperparameters
        self.clip_range_vf = config.clip_range_vf

        # Register PPO-specific metric monitors
        self.metrics_monitor.register_bundle(PPOAlerts(self))

    def losses_for_batch(self, batch, batch_idx):
        # Retrieve immutable PPO parameters
        target_kl = self.config.target_kl

        # use type for this? check sb3
        observations = batch.observations
        actions = batch.actions
        old_logprobs = batch.logprobs
        advantages = batch.advantages
        returns = batch.returns

        # Assert that the tensors are detached
        assert_detached(observations, actions, old_logprobs, advantages, returns)

        # TODO: perform these ops before calling losses_for_batch?
        # Batch-normalize advantage if requested
        advantages, adv_norm_metrics = normalize_batch_with_metrics(
            advantages, self.config.normalize_advantages, "roll/adv"
        )

        # Infer policy_distribution and value_predictions from the actor critic model
        policy_dist, values_pred = self.policy_model(observations)
        if values_pred is None:
            raise ValueError(
                "PPO requires a policy with a value head; set config.policy to 'mlp_actorcritic'."
            )

        # Retrieve the log probabilities of the batch actions under the current policy
        # (since we are training multiple epochs using same rollout, the policy will
        # shift away from the one used to collect the rollout, so log probabilities will change)
        new_logprobs = policy_dist.log_prob(actions)

        # Calculate the ratio of the new policy to the old policy:
        # log(new) - log(old) = log(new/old) => exp(log(new/old)) = new/old = ratio
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Scale the advantages by the change ratio
        scaled_advantages = advantages * ratio

        # Scale the advantages by a clamped ratio that is truncated within the allowed range
        scaled_advantages_clamped = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        # Now ensure that each advantage is the minimum of the two
        # (this ensures that no advantage is beyond the allowed range by truncating 
        # when it passes it and allowing the actual value when its inside the allowed range)
        min_scaled_advantages = torch.min(scaled_advantages, scaled_advantages_clamped)

        # Now we calculate the mean advantage to get the value we want to maximize
        mean_advantage = min_scaled_advantages.mean()

        # Negating the mean advantage give us the policy loss.
        # Gradient descent will minimize the loss, and as a
        # consequence will maximize the mean advantage. Since
        # this advantage is always within a secure range, there
        # will be less variance in the policy loss making optimization more stable.
        policy_loss = -mean_advantage

        # The value head must predict the returns, so its loss is
        # just the MSE between the predicted (values) and target returns
        # NOTE: this must be done in order, second argument must be the target
        # Clip value function updates to prevent large changes (as per PPO paper)
        values_old = batch.values
        values_delta = values_pred - values_old
        v_loss_unclipped = (values_pred - returns) ** 2
        v_clipped = values_old + torch.clamp(
            values_delta,
            -self.clip_range_vf,
            self.clip_range_vf,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        value_loss = v_loss_max.mean()

        # TODO: note down why returns cant be normalized (GAE), but why advantages should be

        # To encourage exploration we can encourage entropy maximization 
        # by making the negative of the current entropy a loss term
        # (this value must be scaled so that it doesn't dominate the 
        # policy loss or value loss and also is not dominated by them)
        entropy = policy_dist.entropy().mean()
        entropy_loss = -entropy

        # TODO: ensure all terms are within similar ranges, add warnings
        # Create the final loss value by mixing the different loss terms
        # according to the coefficients we set in the config 
        # (different weights for each loss term)
        scaled_value_loss = self.vf_coef * value_loss
        scaled_entropy_loss = self.ent_coef * entropy_loss
        loss = policy_loss + scaled_value_loss + scaled_entropy_loss

        # Calculate additional metrics for logging
        # (don't compute gradients during these calculations)
        # NOTE: the reason why a KL early stop would help is that KL divergence measures drift in the whole action distribution, not just the selected actions (like the clipping)
        with torch.no_grad():
            # Measure how many log probs moved beyond the trusted region (average of how many samples are outside the allowed range)
            clip_fraction = ((ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range)).float().mean()

            # Measure how many value predictions were clipped
            clip_fraction_vf = ((values_delta < -self.clip_range_vf) | (values_delta > self.clip_range_vf)).float().mean()

            # TODO: should I make metric explain that explained_var is for value head?
            explained_var = 1 - torch.var(returns - values_pred) / torch.var(returns)

        # Compute KL metrics
        kl_metrics, kl_div, approx_kl = compute_kl_metrics(old_logprobs, new_logprobs)

        # In case the KL divergence exceeded the target, stop training on this epoch
        early_stop_epoch = False
        if target_kl is not None:
            approx_kl_value = float(approx_kl.detach())
            early_stop_epoch = approx_kl_value > target_kl

        metrics = {
            'opt/loss/total': loss.detach(),
            'opt/loss/policy': policy_loss.detach(),
            'opt/loss/entropy': entropy_loss.detach(),
            'opt/loss/entropy_scaled': scaled_entropy_loss.detach(),
            'opt/loss/value': value_loss.detach(),
            'opt/loss/value_scaled': scaled_value_loss.detach(),
            'opt/policy/entropy': entropy.detach(),
            'opt/ppo/clip_fraction': clip_fraction.detach(),
            'opt/ppo/clip_fraction_vf': clip_fraction_vf.detach(),
            'opt/value/explained_var': explained_var.detach(),
            'opt/ppo/kl_stop_triggered': 1 if early_stop_epoch else 0,
            **kl_metrics,
            **adv_norm_metrics,
        }

        # Log all metrics (will be avarages and flushed by end of epoch)
        self.metrics_recorder.record("train", metrics)

        # Return result for training step
        return dict(
            loss=loss,
            early_stop_epoch=early_stop_epoch,
        )
