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
        policy_loss_surrogate_1 = advantages * ratio
        policy_loss_surrogate_2 = advantages * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) # TODO: rename clip_epsilon to clip_range?
        policy_loss = -torch.min(policy_loss_surrogate_1, policy_loss_surrogate_2).mean()
        
        # TODO; what is policy gradient loss to them?
        entropy = dist.entropy().mean()
        entropy_loss = -entropy # TODO: is this entropy loss to them?
        
        
        # Value loss
        # TODO: softcode value scaling coefficient
        #value_loss = 0.5 * ((returns - value_pred) ** 2).mean()
        # TODO: can I use pytoirch util for this?
        # TODO: sb3 is clipping value function as well
        #value_loss = F.mse_loss(rollout_data.returns, values_pred)
        value_loss = ((returns - value_pred) ** 2).mean()
        
        # TODO: detach everything post loss calculation?
        #th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) (do this in optimization step, check how lightning does it)
        # TODO: use single actor critic model
        #value_coef = 1.0
        #loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # Metrics (detached for logging)
        #clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fraction = ((ratio < 1.0 - clip_epsilon) | (ratio > 1.0 + clip_epsilon)).float().mean()
        kl_div = (old_logprobs - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns - value_pred.detach()) / torch.var(returns)

        # TODO: separate concerns... I want to log, but still need to pass loss in order to run backward pass...
        # TODO: log all these stats as /train
        result = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss, # train
            'value_loss': value_loss, # train
            'entropy': entropy.detach(), # train
            'clip_fraction': clip_fraction.detach(), # train
            'kl_div': kl_div.detach(), # train
            'approx_kl': approx_kl.detach(), # train
            'explained_variance': explained_var.detach() # train
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
