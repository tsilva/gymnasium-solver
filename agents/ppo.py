import torch
import torch.nn.functional as F

from utils.policy_factory import build_policy_from_env_and_config
from utils.torch import assert_detached

from .base_agent import BaseAgent


class PPO(BaseAgent):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.clip_range = config.clip_range

    # TODO: do this in init?
    # TODO: call this build models?
    def build_models(self):
        self.policy_model = build_policy_from_env_and_config(self.train_env, self.config)

    def losses_for_batch(self, batch, batch_idx):
        # use type for this? check sb3
        states = batch.observations
        actions = batch.actions
        old_logprobs = batch.log_prob # TODO: call log_probs
        advantages = batch.advantages
        returns = batch.returns

        # Assert that the tensors are detached
        assert_detached(states, actions, old_logprobs, advantages, returns)

        # TODO: use util (use pytorch function if available)
        # TODO: perform these ops before calling losses_for_batch?
        # Batch-normalize advantage if requested
        normalize_advantages = self.config.normalize_advantages == "batch"
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ent_coef = self.config.ent_coef
        vf_coef = self.config.vf_coef

        # Infer policy_distribution and value_predictions from the actor critic model
        policy_dist, values_pred = self.policy_model(states)

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
        value_loss = F.mse_loss(values_pred, returns) # TODO: what is this scale?

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
        scaled_value_loss = vf_coef * value_loss
        scaled_entropy_loss = ent_coef * entropy_loss
        loss = policy_loss + scaled_value_loss + scaled_entropy_loss

        # Calculate additional metrics for logging
        # (don't compute gradients during these calculations)
        with torch.no_grad():
            # Measure how many log probs moved beyond the trusted region (average of how many samples are outside the allowed range)
            clip_fraction = ((ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range)).float().mean()


            kl_div = (old_logprobs - new_logprobs).mean()
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            explained_var = 1 - torch.var(returns - values_pred) / torch.var(returns)

        # Log all metrics (will be avarages and flushed by end of epoch)
        self.metrics.record_train({
            'loss': loss.detach(),
            'policy_loss': policy_loss.detach(),
            'entropy_loss': entropy_loss.detach(),
            'entropy_loss_scaled': scaled_entropy_loss.detach(),
            'value_loss': value_loss.detach(),
            'value_loss_scaled': scaled_value_loss.detach(),
            'entropy': entropy.detach(),
            'clip_fraction': clip_fraction.detach(),
            'kl_div': kl_div.detach(),
            'approx_kl': approx_kl.detach(),
            'explained_variance': explained_var.detach()
        })

        return loss
    
    # TODO: should schedulers be callbacks?
    # TODO: find a way to not have to inherit this
    def _update_schedules(self):
        super()._update_schedules()
        self._update_schedules__clip_range()

    def _update_schedules__clip_range(self):
        if self.config.clip_range_schedule != 'linear': return

        progress = self._calc_training_progress()
        new_clip_range = max(self.config.clip_range * (1.0 - progress), 0.0)
        self.clip_range = new_clip_range

        # Log scheduled clip_range under train namespace
        self.metrics.record_train({
            'clip_range': new_clip_range
        })
