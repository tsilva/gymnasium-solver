import torch
from torch.distributions import Categorical
from .base import Learner

class PPOLearner(Learner):
    """PPO-specific agent implementation with optional shared backbone support"""
    
    def __init__(self, config, build_env_fn, rollout_collector, policy_model, value_model=None, shared_model=None):
        super().__init__(config, build_env_fn, rollout_collector, policy_model, value_model=value_model)
        self.save_hyperparameters(ignore=['build_env_fn', 'rollout_collector', 'policy_model', 'value_model', 'shared_model'])
        
        self.policy_model = policy_model
        self.value_model = value_model
        self.shared_model = shared_model
        self.use_shared_backbone = shared_model is not None

        self.ppo_loss = PPOLoss(config.clip_epsilon, config.entropy_coef, self.use_shared_backbone)

    def compute_loss(self, batch):
        """Compute PPO-specific losses"""
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch
        
        if self.use_shared_backbone:
            return self.ppo_loss.compute(
                states, actions, old_logps, advantages, returns, 
                self.shared_model, None, use_shared=True
            )
        else:
            return self.ppo_loss.compute(
                states, actions, old_logps, advantages, returns, 
                self.policy_model, self.value_model, use_shared=False
            )
        
    def optimize_models(self, loss_results):
        """Optimize PPO policy and value models"""
        if self.use_shared_backbone:
            # Single optimizer for shared backbone
            opt_shared = self.optimizers()
            opt_shared.zero_grad()
            self.manual_backward(loss_results['total_loss'])
            opt_shared.step()
        else:
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
        if self.use_shared_backbone:
            return torch.optim.Adam(self.shared_model.parameters(), lr=self.config.policy_lr)
        else:
            return [
                torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr),
                torch.optim.Adam(self.value_model.parameters(), lr=self.config.value_lr)
            ]

    def forward(self, x):
        if self.use_shared_backbone:
            return self.shared_model(x)
        else:
            return self.policy_model(x)

class PPOLoss:
    def __init__(self, clip_epsilon, entropy_coef, use_shared_backbone=False):
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.use_shared_backbone = use_shared_backbone
    
    def compute(self, states, actions, old_logps, advantages, returns, policy_model, value_model=None, use_shared=False):
        if use_shared:
            # Shared backbone: get both policy and value in one forward pass
            logits, value_pred = policy_model.forward_both(states)
            value_pred = value_pred.squeeze()
        else:
            # Separate models
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
        
        # For shared backbone, we need a total loss for the single optimizer
        if use_shared:
            result['total_loss'] = policy_loss + value_loss
            
        return result
    