import torch
from torch.distributions import Categorical
from .base import Learner

class REINFORCELearner(Learner):
    """REINFORCE-specific agent implementation"""
    def __init__(self, config, build_env_fn, rollout_collector, policy_model):
        super().__init__(config, build_env_fn, rollout_collector, policy_model)
        self.save_hyperparameters(ignore=['build_env_fn', 'rollout_collector', 'policy_model', 'value_model'])

        self.policy_model = policy_model

        # Create REINFORCE-specific models and components
        self.reinforce_loss = REINFORCELoss(config.entropy_coef)

    def compute_loss(self, batch):
        """Compute REINFORCE-specific losses"""
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch
        
        # REINFORCE uses Monte Carlo returns directly (not advantages)
        return self.reinforce_loss.compute(
            states, actions, returns, self.policy_model
        )
        
    def optimize_models(self, loss_results):
        """Optimize REINFORCE policy model"""
        optimizer = self.optimizers()
        
        # Optimize policy
        optimizer.zero_grad()
        self.manual_backward(loss_results['policy_loss'])
        optimizer.step()

    def configure_optimizers(self):
        # REINFORCE only needs policy optimizer
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr)

    def forward(self, x):
        return self.policy_model(x)
    
class REINFORCELoss:
    def __init__(self, entropy_coef):
        self.entropy_coef = entropy_coef
    
    def compute(self, states, actions, returns, policy_model):
        # Policy loss using REINFORCE (policy gradient with Monte Carlo returns)
        logits = policy_model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # TODO: why does this work with returns detached?
        # Ensure returns are detached from any previous computation graph
        returns_detached = returns.detach()
        
        # REINFORCE loss: -log_prob * return (negative because we want to maximize)
        policy_loss = -(log_probs * returns_detached).mean() - self.entropy_coef * entropy
        
        # Metrics (detached for logging)
        return {
            'policy_loss': policy_loss,
            'entropy': entropy.detach(),
            'log_prob_mean': log_probs.mean().detach(),
            'returns_mean': returns.mean().detach()
        }
