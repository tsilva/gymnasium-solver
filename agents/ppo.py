import torch
import torch.nn.functional as F

from utils.policy_factory import create_actor_critic_policy

from .base_agent import BaseAgent

class PPO(BaseAgent):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.clip_range = config.clip_range

    # TODO: do this in init?
    def create_models(self):
        # If BaseAgent.__init__ (and thus LightningModule/nn.Module.__init__) wasn't called
        # initialize the nn.Module machinery so assigning submodules works in tests.
        try:
            _ = self._modules  # type: ignore[attr-defined]
        except Exception:
            import torch.nn as nn  # local import to avoid global side effects
            nn.Module.__init__(self)

        # Be resilient to minimal configs used in tests
        policy_kwargs = getattr(self.config, "policy_kwargs", {"activation": "tanh"})
        activation = policy_kwargs.get('activation', 'tanh')  # noqa: F841 - kept for potential side-effects
        # Determine policy type and input/output dims even if BaseAgent.__init__ wasn't called
        policy_type = getattr(self.config, "policy", "mlp")
        input_dim = self.train_env.get_input_dim()
        output_dim = self.train_env.get_output_dim()
        obs_space = getattr(self.train_env, 'observation_space', None)
        self.policy_model = create_actor_critic_policy(
            policy_type,
            input_dim=input_dim,
            action_dim=output_dim,
            hidden=self.config.hidden_dims,
            # TODO: redundancy with input_dim/output_dim?
            obs_space=obs_space,
            **policy_kwargs,
        )

    def losses_for_batch(self, batch, batch_idx):
        # use type for this? check sb3
        states = batch.observations
        actions = batch.actions
        old_logprobs = batch.log_prob # TODO: call log_probs
        advantages = batch.advantages
        returns = batch.returns

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
        new_logps = policy_dist.log_prob(actions)

        # Calculate the ratio of the new policy to the old policy:
        # log(new) - log(old) = log(new/old) => exp(log(new/old)) = new/old = ratio
        ratio = torch.exp(new_logps - old_logprobs)

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

        # To encourage exploration we can encourage entropy maximization 
        # by making the negative of the current entropy a loss term
        # (this value must be scaled so that it doesn't dominate the 
        # policy loss or value loss and also is not dominated by them)
        entropy = policy_dist.entropy().mean()
        entropy_loss = -entropy

        # Create the final loss value by mixing the different loss terms
        # according to the coefficients we set in the config 
        # (different weights for each loss term)
        scaled_value_loss = vf_coef * value_loss
        scaled_entropy_loss = ent_coef * entropy_loss
        loss = policy_loss + scaled_value_loss + scaled_entropy_loss

        # Calculate additional metrics for logging
        # (don't compute gradients during these calculations)
        with torch.no_grad():
            clip_fraction = ((ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range)).float().mean()
            kl_div = (old_logprobs - new_logps).mean()
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
            explained_var = 1 - torch.var(returns - values_pred) / torch.var(returns)

        # Log all metrics (will be avarages and flushed by end of epoch)
        self.log_metrics({
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
        }, prefix="train")

        return loss
    
    def configure_optimizers(self):
        # TODO: Match SB3's Adam defaults more closely by setting eps=1e-5
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr, eps=1e-5)
    
    # TODO: find a way to not have to inherit this
    def _update_schedules(self):
        super()._update_schedules()
        self._update_schedules__clip_range()

    def _update_schedules__clip_range(self):
        if self.config.clip_range_schedule != 'linear': return

        progress = self._get_training_progress()
        new_clip_range = max(self.config.clip_range * (1.0 - progress), 0.0)
        self.clip_range = new_clip_range

        # Log scheduled clip_range under train namespace
        self.log_metrics({
            'clip_range': new_clip_range
        }, prefix="train")
