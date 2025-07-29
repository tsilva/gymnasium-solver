import torch
from torch.distributions import Categorical
from .base_agent import BaseAgent
from utils.models import ActorCritic
from utils.misc import prefix_dict_keys

class REINFORCE(BaseAgent):
    
    # TODO: add support for policy only model
    def create_models(self):
        self.policy_model = ActorCritic(
            self.input_dim,
            self.output_dim, # TODO: should be called obs_dim and act_dim?
            hidden=self.config.hidden_dims
        )
    
    def train_on_batch(self, batch, batch_idx):
        states = batch.observations
        actions = batch.actions
        returns = batch.returns
        
        ent_coef = self.config.ent_coef

        dist, _ = self.policy_model(states)
        log_probs = dist.log_prob(actions)
      
        policy_loss = -(log_probs * returns).mean()

        entropy = dist.entropy().mean()
        entropy_loss = -entropy
        
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
