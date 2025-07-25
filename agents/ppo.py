import torch
import torch.nn.functional as F
from .base_agent import BaseAgent
from utils.models import ActorCritic
from utils.misc import prefix_dict_keys

class PPO(BaseAgent):

    def create_models(self):
        # TODO: should I call this policy model?
        self.policy_model = ActorCritic(
            self.input_dim,
            self.output_dim, # TODO: should be called obs_dim and act_dim?
            hidden=self.config.hidden_dims
        )
    
    def training_step(self, batch, batch_idx):
        # use type for this? check sb3
        states, actions, rewards, dones, old_logprobs, values, advantages, returns, frames = batch
        
        clip_range = self.config.clip_range
        ent_coef = self.config.ent_coef
        val_coef = self.config.val_coef

        dist, value = self.policy_model(states)
    
        new_logps = dist.log_prob(actions)

        ratio = torch.exp(new_logps - old_logprobs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) # TODO: rename clip_range to clip_range?
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # TODO; what is policy gradient loss to them?
        entropy = dist.entropy().mean()
        entropy_loss = -entropy # TODO: is this entropy loss to them?
        
        # Value loss
        # TODO: sb3 is clipping value function as well
        value_loss = ((returns - value) ** 2).mean()
        value_loss_2 = F.mse_loss(returns, value)
        
        # TODO: detach everything post loss calculation?
        #th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm) (do this in optimization step, check how lightning does it)
        loss = policy_loss + (val_coef * value_loss) + (ent_coef * entropy_loss)

        # Metrics (detached for logging)
        #clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fraction = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean()
        kl_div = (old_logprobs - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns - value.detach()) / torch.var(returns)

        metrics = prefix_dict_keys({
            'loss' : loss.detach().item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'value_loss': value_loss.item(), 
            'entropy': entropy.detach().item(), 
            'clip_fraction': clip_fraction.detach().item(), 
            'kl_div': kl_div.detach().item(), 
            'approx_kl': approx_kl.detach().item(), 
            'explained_variance': explained_var.detach().item()
        }, "train")
        self.log_metrics(metrics)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr)
