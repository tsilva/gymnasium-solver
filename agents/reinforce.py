import torch

from utils.models import PolicyOnly, CNNPolicyOnly

from .base_agent import BaseAgent


class REINFORCE(BaseAgent):
    
    def create_models(self):
        input_dim = self.train_env.get_input_dim()
        output_dim = self.train_env.get_output_dim()
        policy_type = getattr(self.config, 'policy', 'MlpPolicy')
        policy_kwargs = getattr(self.config, 'policy_kwargs', None) or {}
        activation = getattr(self.config, 'activation', 'tanh')

        if isinstance(policy_type, str) and policy_type.lower() == 'cnnpolicy':
            obs_space = self.train_env.observation_space
            obs_shape = getattr(obs_space, 'shape', None)
            if obs_shape is None:
                raise ValueError("CNNPolicy requires Box observation space with shape")
            if len(obs_shape) == 3:
                hwc = (obs_shape[0], obs_shape[1], obs_shape[2])
            elif len(obs_shape) == 2:
                hwc = (obs_shape[0], obs_shape[1], 1)
            else:
                side = int(max(input_dim, 1) ** 0.5)
                hwc = (side, side, 1)
            self.policy_model = CNNPolicyOnly(
                obs_shape=hwc,
                action_dim=output_dim,
                hidden=self.config.hidden_dims,
                activation=activation,
                **policy_kwargs,
            )
        else:
            self.policy_model = PolicyOnly(
                input_dim,
                output_dim,
                hidden=self.config.hidden_dims,
                activation=activation,
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
