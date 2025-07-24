import torch
from torch.distributions import Categorical
from .base_agent import BaseAgent
from utils.models import PolicyNet, ValueNet

class PPO(BaseAgent):

    def create_models(self):
        input_dim = self.config.env_spec['input_dim']
        output_dim = self.config.env_spec['output_dim']
        self.policy_model = PolicyNet(input_dim, output_dim, self.config.hidden_dims)
        self.value_model = ValueNet(input_dim, self.config.hidden_dims)

    def compute_loss(self, batch):
        # use type for this? check sb3
        states, actions, rewards, dones, old_logprobs, values, advantages, returns, frames = batch
        
        clip_epsilon = self.config.clip_epsilon
        entropy_coef = self.config.entropy_coef

        logits = self.policy_model(states)
        value_pred = self.value_model(states).squeeze()
    
        dist = Categorical(logits=logits)
        new_logps = dist.log_prob(actions)

        ratio = torch.exp(new_logps - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        entropy = dist.entropy().mean()
        
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
        
        # Value loss
        # TODO: softcode value scaling coefficient
        #value_loss = 0.5 * ((returns - value_pred) ** 2).mean()
        # TODO: can I use pytoirch util for this?
        value_loss = ((returns - value_pred) ** 2).mean()
        
        # TODO: detach everything post loss calculation?

        # Metrics (detached for logging)
        clip_fraction = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float().mean()
        kl_div = (old_logprobs - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns - value_pred.detach()) / torch.var(returns)

        # TODO: separate concerns... I want to log, but still need to pass loss in order to run backward pass...
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
