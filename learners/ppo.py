import torch
from torch.distributions import Categorical
from .base import Learner

class PPOLearner(Learner):
    """PPO-specific agent implementation with optional shared backbone support"""
    
    def __init__(self, config, build_env_fn, rollout_collector, policy_model, value_model=None):
        super().__init__(config, build_env_fn, rollout_collector, policy_model, value_model=value_model)
        self.save_hyperparameters(ignore=['build_env_fn', 'rollout_collector', 'policy_model', 'value_model'])
        
        self.policy_model = policy_model
        self.value_model = value_model
        
        self.ppo_loss = PPOLoss(config.clip_epsilon, config.entropy_coef)

    def compute_loss(self, batch):
        """Compute PPO-specific losses"""
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch
        
        return self.ppo_loss.compute(
            states, actions, old_logps, advantages, returns, 
            self.policy_model, self.value_model
        )
        
    def optimize_models(self, loss_results):
        """Optimize PPO policy and value models"""
        # Separate optimizers for policy and value
        opt_policy, opt_value = self.optimizers()
        
        # Optimize policy
        opt_policy.zero_grad()
        self.manual_backward(loss_results['policy_loss'])
        opt_policy.step()

        # Optimize value function
        opt_value.zero_grad()
        self.manual_backward(loss_results['value_loss'])
        opt_value.step()

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr),
            torch.optim.Adam(self.value_model.parameters(), lr=self.config.value_lr)
        ]

    def forward(self, x):
        return self.policy_model(x)
    
    def on_train_epoch_start(self):
        """Override to handle shared backbone model updates"""
        self.metrics.reset()
        
        # For separate models, use the original approach
        self.rollout_collector.update_models(
            self.policy_model.state_dict(), 
            self.value_model.state_dict() if self.value_model else None
        )
        
        # Collect new rollout if needed
        if (self.current_epoch + 1) % self.config.rollout_interval == 0:
            self._collect_and_update_rollout()

class PPOLoss:
    def __init__(self, clip_epsilon, entropy_coef):
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
    
    def compute(self, states, actions, old_logps, advantages, returns, policy_model, value_model=None):
        logits = policy_model(states)
        value_pred = value_model(states).squeeze()
    
        dist = Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        
        # Detach tensors that come from rollout data to prevent in-place operation errors
        old_logps_detached = old_logps.detach()
        advantages_detached = advantages.detach()
        returns_detached = returns.detach()
        
        ratio = torch.exp(new_logps - old_logps_detached)
        surr1 = ratio * advantages_detached
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_detached
        entropy = dist.entropy().mean()
        
        policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        
        # Value loss
        value_loss = 0.5 * ((returns_detached - value_pred) ** 2).mean()
        
        # Metrics (detached for logging)
        clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
        kl_div = (old_logps_detached - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns_detached - value_pred.detach()) / torch.var(returns_detached)
        
        result = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy.detach(),
            'clip_fraction': clip_fraction.detach(),
            'kl_div': kl_div.detach(),
            'approx_kl': approx_kl.detach(),
            'explained_var': explained_var.detach()
        }
        
        return result
    