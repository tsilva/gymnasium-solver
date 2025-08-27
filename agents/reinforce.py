from utils.policy_factory import create_policy

from .base_agent import BaseAgent

import torch

def assert_detached(*tensors: torch.Tensor):
    for t in tensors:
        assert not t.requires_grad, "Tensor still requires grad"
        assert t.grad_fn is None, "Tensor is still connected to a computation graph"
    return True
    
class REINFORCE(BaseAgent):
    
    def create_models(self):
        input_dim = self.train_env.get_input_dim()
        output_dim = self.train_env.get_output_dim()
        policy_kwargs = self.config.policy_kwargs

        self.policy_model = create_policy(
            self.config.policy,
            input_dim=input_dim,
            action_dim=output_dim,
            hidden_dims=self.config.hidden_dims,
            obs_space=getattr(self.train_env, 'observation_space', None),  # TODO: what is observation space being used for
            **policy_kwargs,
        )
    
    def rollout_collector_hyperparams(self):
        # Override to disable GAE for REINFORCE - use pure Monte Carlo returns
        base_params = self.config.rollout_collector_hyperparams()
        base_params['use_gae'] = False # TODO: what is this for?
        return base_params
    
    def losses_for_batch(self, batch, batch_idx):
        # Retrieve tensors from batch
        states = batch.observations
        actions = batch.actions
        returns = batch.returns

        # Assert that the tensors are detached
        assert_detached(states, actions, returns)
        
        # Calculate the policy targets to scale the log probabilities by;
        # - returns: unscaled returns (vanilla REINFORCE)
        # - batch_normalized_returns: returns normalized by batch mean and std (reduces variance)
        # - advantages: REINFORCE with baseline (reduces variance in a more informed way)
        policy_targets = returns
        normalize_returns = self.config.normalize_returns
        if normalize_returns == "batch": 
            batch_normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            policy_targets = batch_normalized_returns
        elif normalize_returns == "baseline": # TODO: remove use_baseline
            advantages = (returns - baseline).detach() # TODO: why detach?
            policy_targets = advantages

        # Get the log probabilities for each 
        # action given the current state
        dist, _ = self.policy_model(states)
        log_probs = dist.log_prob(actions)

        # Scale each action log probability by the monte carlo return for that state;
        # this will boost confident action that lead to high returns more than inconfident 
        # actions tthat lead to high returns or confident actions that lead to low returns 
        # or incofident actions that lead to low returns; after this calculate the mean
        # to get a single scalar value that represents how good the policy is; this is the
        # value we want to increase, the bigger it gets the better the policy is, because
        # the highest loss value is achieved when the best returns were achieved with maximum confidence
        # (note: we negate the calculation because pytorch optimizers minimize loss, so by minimizing 
        # the value we want to increase, we are maximizing it)
        # Scale each action probability 
        policy_loss = -(log_probs * policy_targets).mean()
        
        # Calculate how far the distribution is from being uniform;
        # the less uniform it is the less randomness there is in the policy,
        # meaning there is less exploration, we want to both maximize the policy 
        # returns while avoiding collapsing entropy, which would stop exploration 
        # of new actions, preventing further learning
        entropy = dist.entropy().mean()
        entropy_loss = -entropy
        
        # The final loss is the sum of the policy loss and the entropy loss;
        # the higher the entropy coefficient the more priority we give to exploration
        ent_coef = self.config.ent_coef
        loss = policy_loss + (ent_coef * entropy_loss)
        
        # Log the metrics for monitoring training progress
        self.log_metrics({
            'loss' : loss.detach(),
            'policy_loss': policy_loss.detach(),
            'entropy_loss': entropy_loss.detach(),
            'entropy': entropy.detach()
        }, prefix="train")
        return loss

