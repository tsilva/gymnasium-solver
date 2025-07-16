import torch
from torch.distributions import Categorical
from .base import Learner

class PPOLearner(Learner):
    """PPO-specific agent implementation"""
    
    def __init__(self, config, build_env_fn, rollout_collector, policy_model, value_model):
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

class PPOLoss:
    def __init__(self, clip_epsilon, entropy_coef):
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
    
    def compute(self, states, actions, old_logps, advantages, returns, policy_model, value_model):
        # Policy loss
        logits = policy_model(states)
        dist = Categorical(logits=logits)
        new_logps = dist.log_prob(actions)
        
        ratio = torch.exp(new_logps - old_logps)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        entropy = dist.entropy().mean()
        
        policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        
        # Value loss
        value_pred = value_model(states).squeeze()
        value_loss = 0.5 * ((returns - value_pred) ** 2).mean()
        
        # Metrics
        clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
        kl_div = (old_logps - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns - value_pred) / torch.var(returns)
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'clip_fraction': clip_fraction,
            'kl_div': kl_div,
            'approx_kl': approx_kl,
            'explained_var': explained_var
        }
    