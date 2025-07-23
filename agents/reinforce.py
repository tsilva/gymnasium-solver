import torch
from torch.distributions import Categorical
from .base_agent import BaseAgent

class REINFORCE(BaseAgent):
    
    def compute_loss(self, batch):
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch
        
        # REINFORCE uses Monte Carlo returns dir
        # Policy loss using REINFORCE (policy gradient with Monte Carlo returns)
        logits = self.policy_model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # REINFORCE loss: -log_prob * return (negative because we want to maximize)
        policy_loss = -(log_probs * returns).mean() - self.config.entropy_coef * entropy
        
        # Metrics (detached for logging)
        return {
            'policy_loss': policy_loss,
            'entropy': entropy.detach(),
            'log_prob_mean': log_probs.mean().detach(),
            'returns_mean': returns.mean().detach()
        }
        
    def optimize_models(self, loss_results):
        """Optimize REINFORCE policy model"""
        optimizer = self.optimizers()
        
        # Optimize policy
        optimizer.zero_grad()
        self.manual_backward(loss_results['policy_loss'])
        optimizer.step()

    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr)

    def forward(self, x):
        return self.policy_model(x)
    