import torch
from torch.distributions import Categorical
from .base_agent import BaseAgent
from utils.models import PolicyNet
from utils.misc import prefix_dict_keys

class REINFORCE(BaseAgent):

    def create_models(self):
        input_dim = self.config.env_spec['input_dim']
        output_dim = self.config.env_spec['output_dim']
        self.policy_model = PolicyNet(input_dim, output_dim, self.config.hidden_dims)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch
        
        ent_coef = self.config.ent_coef

        # REINFORCE uses Monte Carlo returns dir
        # Policy loss using REINFORCE (policy gradient with Monte Carlo returns)
        logits = self.policy_model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * returns).mean()

        # TODO; what is policy gradient loss to them?
        entropy = dist.entropy().mean()
        entropy_loss = -entropy # TODO: is this entropy loss to them?
        
        loss = policy_loss + (ent_coef * entropy_loss)
        
        metrics = prefix_dict_keys({
            'loss' : loss.detach().item(),
            'policy_loss': policy_loss.detach().item(),
            'entropy_loss': entropy_loss.detach().item(),
            'entropy': entropy.detach().item(), 
        }, "train")
        self.log_metrics(metrics)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr)
