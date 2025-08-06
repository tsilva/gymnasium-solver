import torch
from .base_agent import BaseAgent
from utils.models import PolicyOnly
from utils.misc import prefix_dict_keys

class REINFORCE(BaseAgent):
    
    def create_models(self):
        input_dim = self.train_env.get_input_dim()
        output_dim = self.train_env.get_output_dim()
        self.policy_model = PolicyOnly(
            input_dim,
            output_dim,
            hidden=self.config.hidden_dims
        )
    
    def rollout_collector_hyperparams(self):
        # Override to disable GAE for REINFORCE - use pure Monte Carlo returns
        base_params = self.config.rollout_collector_hyperparams()
        base_params['use_gae'] = False
        return base_params
    
    def train_on_batch(self, batch, batch_idx):
        states = batch.observations
        actions = batch.actions
        returns = batch.returns
        advantages = batch.advantages
        
        ent_coef = self.config.ent_coef
        use_baseline = getattr(self.config, 'use_baseline', False)

        dist, _ = self.policy_model(states)
        log_probs = dist.log_prob(actions)
      
        # Choose between returns (vanilla REINFORCE) or advantages (REINFORCE with baseline)
        if use_baseline:
            # Use advantages (returns - baseline) for baseline subtraction
            policy_targets = advantages
            
            # Batch-level advantage normalization if enabled
            if getattr(self.config, 'advantage_norm', 'batch') == "batch" and len(policy_targets) > 1:
                policy_targets = (policy_targets - policy_targets.mean()) / (policy_targets.std() + 1e-8)
        else:
            # Use raw returns for vanilla REINFORCE
            policy_targets = returns
            
        policy_loss = -(log_probs * policy_targets).mean()

        entropy = dist.entropy().mean()
        entropy_loss = -entropy
        
        loss = policy_loss + (ent_coef * entropy_loss)
        
        self.log_metrics({
            'loss' : loss.detach().item(),
            'policy_loss': policy_loss.detach().item(),
            'entropy_loss': entropy_loss.detach().item(),
            'entropy': entropy.detach().item()
        }, prefix="train")
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.policy_model.parameters(), lr=self.config.policy_lr)
