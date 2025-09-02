from utils.policy_factory import create_policy

from .base_agent import BaseAgent

import torch

# TODO: move to util
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

    # TODO: only does something with normalization off, but even that way it doesnt converge
    def losses_for_batch(self, batch, batch_idx):
        # Retrieve tensors from batch
        states = batch.observations
        actions = batch.actions
        old_logprobs = batch.log_prob
        returns = batch.returns
        advantages = batch.advantages

        # Assert that the tensors are detached
        assert_detached(states, actions, returns, advantages)
        
        # Normalize returns if requested
        if self.config.normalize_returns == "batch": 
            # TODO: use util
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Normalize advantages if requested
        if self.config.normalize_advantages == "batch": 
            # TODO: use util
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Pick the configured policy targets
        if self.config.policy_targets == "returns": 
            policy_targets = returns
        elif self.config.policy_targets == "advantages": 
            policy_targets = advantages
        else: 
            raise ValueError(f"Invalid policy targets: {self.config.policy_targets}")

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

        # KL diagnostics between rollout policy (old) and current policy (new)
        # Matches PPO-style on-action KL estimates
        with torch.no_grad():
            ratio = torch.exp(log_probs - old_logprobs)
            kl_div = (old_logprobs - log_probs).mean()
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        
        # The final loss is the sum of the policy loss and the entropy loss;
        # the higher the entropy coefficient the more priority we give to exploration
        ent_coef = self.config.ent_coef
        loss = policy_loss + (ent_coef * entropy_loss)
        
        # Log the metrics for monitoring training progress
        self.log_metrics({
            'loss' : loss.detach(),
            'policy_loss': policy_loss.detach(),
            'entropy_loss': entropy_loss.detach(),
            'entropy': entropy.detach(),
            'kl_div': kl_div.detach(),
            'approx_kl': approx_kl.detach(),
            'policy_targets_mean': policy_targets.mean().detach(),
            'policy_targets_std': policy_targets.std().detach()
        }, prefix="train")
        return loss
