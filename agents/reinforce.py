import torch

from utils.models import PolicyOnly

from .base_agent import BaseAgent


class REINFORCE(BaseAgent):
    
    def create_models(self):
        input_dim = self.train_env.get_input_dim()
        output_dim = self.train_env.get_output_dim()
        self.policy_model = PolicyOnly(
            input_dim,
            output_dim,
            hidden=self.config.hidden_dims,
            activation=self.config.activation,
        )
    
    def rollout_collector_hyperparams(self):
        # Override to disable GAE for REINFORCE - use pure Monte Carlo returns
        base_params = self.config.rollout_collector_hyperparams()
        base_params['use_gae'] = False
        return base_params
    
    def losses_for_batch(self, batch, batch_idx):
        states = batch.observations
        actions = batch.actions
        returns = batch.returns
        advantages = batch.advantages
        
        ent_coef = self.config.ent_coef
        use_baseline = self.config.use_baseline
        normalize_advantages = self.config.normalize_advantages == "batch"

        dist, _ = self.policy_model(states)
        log_probs = dist.log_prob(actions)
      
        policy_targets = returns
        
        # Choose between returns (vanilla REINFORCE) or advantages (REINFORCE with baseline)
        if use_baseline:
            policy_targets = advantages
            # TODO: call this normalize_advantages?
            # TODO: call this policy_targets?
            if normalize_advantages: policy_targets = (policy_targets - policy_targets.mean()) / (policy_targets.std() + 1e-8)
       
        policy_loss = -(log_probs * policy_targets).mean()

        entropy = dist.entropy().mean()
        entropy_loss = -entropy
        
        loss = policy_loss + (ent_coef * entropy_loss)
        
        self.log_metrics({
            'loss' : loss.detach(),
            'policy_loss': policy_loss.detach(),
            'entropy_loss': entropy_loss.detach(),
            'entropy': entropy.detach()
        }, prefix="train")
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr)
