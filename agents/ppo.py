import torch
import torch.nn.functional as F
from .base_agent import BaseAgent
from utils.models import ActorCritic

class PPO(BaseAgent):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.clip_range = config.clip_range

    # TODO: do this in init?
    def create_models(self):
        input_dim = self.train_env.get_input_dim()
        output_dim = self.train_env.get_output_dim()
        self.policy_model = ActorCritic(
            input_dim,
            output_dim, # TODO: should be called obs_dim and act_dim?
            hidden=self.config.hidden_dims
        )
    
    def train_on_batch(self, batch, batch_idx):
        # use type for this? check sb3
        states = batch.observations
        actions = batch.actions
        old_logprobs = batch.old_log_prob
        advantages = batch.advantages
        returns = batch.returns

        normalize_advantages = self.config.normalize_advantages == "batch"

        # Batch-level advantage normalization if enabled
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ent_coef = self.config.ent_coef
        vf_coef = self.config.vf_coef

        dist, value = self.policy_model(states)

        new_logps = dist.log_prob(actions)

        ratio = torch.exp(new_logps - old_logprobs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Entropy bonus
        entropy = dist.entropy().mean()
        entropy_loss = -entropy

        # Value loss (predictions first, targets second)
        # NOTE: this must be done in order, second argument must be the target
        value_loss = F.mse_loss(value, returns)

        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

        # Metrics (detached for logging)
        clip_fraction = ((ratio < 1.0 - self.clip_range) | (ratio > 1.0 + self.clip_range)).float().mean()
        kl_div = (old_logprobs - new_logps).mean()
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        explained_var = 1 - torch.var(returns - value.detach()) / torch.var(returns)

        self.log_metrics({
            'loss': loss.detach().item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.detach().item(),
            'clip_fraction': clip_fraction.detach().item(),
            'kl_div': kl_div.detach().item(),
            'approx_kl': approx_kl.detach().item(),
            'explained_variance': explained_var.detach().item()
        }, prefix="train")

        return loss
    
    def configure_optimizers(self):
        # TODO: Match SB3's Adam defaults more closely by setting eps=1e-5
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr, eps=1e-5)
    
    # TODO: find a way to not have to inherit this
    def _update_schedules(self):
        super()._update_schedules()
        self._update_schedules__clip_range()

    def _update_schedules__clip_range(self):
        if self.config.clip_range_schedule != 'linear': return

        progress = self._get_training_progress()
        new_clip_range = max(self.config.clip_range * (1.0 - progress), 0.0)
        self.clip_range = new_clip_range

        # TODO: this should not be logged here
        self.log_metrics({
            'clip_range': new_clip_range    
        }, prefix="train")
