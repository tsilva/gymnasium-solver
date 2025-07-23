import torch
from torch.distributions import Categorical
from .base import Learner

class PPOLearner(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # TODO: review how save_hyperparameters works
        self.save_hyperparameters(ignore=['build_env_fn', 'train_rollout_collector', 'policy_model', 'value_model', 'eval_rollout_collector'])

    def compute_loss(self, batch):
        states, actions, rewards, dones, old_logprobs, values, advantages, returns, frames = batch
        
        logits = self.policy_model(states)
        value_pred = self.value_model(states).squeeze()
    
        dist = Categorical(logits=logits)
        new_logps = dist.log_prob(actions)

        ratio = torch.exp(new_logps - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        entropy = dist.entropy().mean()
        
        policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        
        # Value loss
        # TODO: softcode value scaling coefficient
        value_loss = 0.5 * ((returns - value_pred) ** 2).mean()
        
        # Metrics (detached for logging)
        clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
        kl_div = (old_logprobs - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns - value_pred.detach()) / torch.var(returns)

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
