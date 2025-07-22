import torch
from torch.distributions import Categorical
from .base import Learner

class PPOLearner(Learner):
    """PPO-specific agent implementation with optional shared backbone support"""
    
    def __init__(self, config, train_rollout_collector, policy_model, value_model=None, eval_rollout_collector=None):
        super().__init__(config, train_rollout_collector, policy_model, value_model=value_model, eval_rollout_collector=eval_rollout_collector)
        
        # TODO: review how save_hyperparameters works
        self.save_hyperparameters(ignore=['train_rollout_collector', 'policy_model', 'value_model', 'eval_rollout_collector'])
        
        self.policy_model = policy_model
        self.value_model = value_model
        
        self.ppo_loss = PPOLoss(config.clip_epsilon, config.entropy_coef)

    def compute_loss(self, batch):
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch
        
        return self.ppo_loss.compute(
            states, actions, old_logps, advantages, returns, 
            self.policy_model, self.value_model
        )
        
    def optimize_models(self, loss_results):
        """Optimize PPO policy and value models"""
        # Separate optimizers for policy and value
        policy_optimizer, value_optimizer = self.optimizers()
        
        # Optimize policy
        policy_optimizer.zero_grad()
        self.manual_backward(loss_results['policy_loss'])
        policy_optimizer.step()

        # Optimize value function
        value_optimizer.zero_grad()
        self.manual_backward(loss_results['value_loss'])
        value_optimizer.step()

    def configure_optimizers(self):
        return [
            torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr),
            torch.optim.Adam(self.value_model.parameters(), lr=self.config.value_lr)
        ]


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
        old_logps_detached = old_logps.detach() # TODO: aren't these already detached?
        advantages_detached = advantages.detach() # TODO: aren't these already detached?
        returns_detached = returns.detach() # TODO: aren't these already detached?
        
        ratio = torch.exp(new_logps - old_logps_detached)
        surr1 = ratio * advantages_detached
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_detached
        entropy = dist.entropy().mean()
        
        policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        
        # Value loss
        # TODO: softcode value scaling coefficient
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
    