import torch

from utils.policy_factory import build_policy
from utils.torch import assert_detached

from .base_agent import BaseAgent


class REINFORCE(BaseAgent):
    
    def build_models(self):
        input_shape = self.train_env.observation_space.shape
        output_shape = self.train_env.action_space.shape
        if not output_shape: output_shape = (self.train_env.action_space.n,)
        self.policy_model = build_policy(
            self.config.policy,
            input_shape=input_shape,
            hidden_dims=self.config.hidden_dims,
            output_shape=output_shape,
            activation=self.config.activation,
            **self.config.policy_kwargs,
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
        logprobs = dist.log_prob(actions)

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
        policy_loss = -(logprobs * policy_targets).mean()
        
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
            ratio = torch.exp(logprobs - old_logprobs)
            kl_div = (old_logprobs - logprobs).mean()
            approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
        
        # The final loss is the sum of the policy loss and the entropy loss;
        # the higher the entropy coefficient the more priority we give to exploration
        ent_coef = self.config.ent_coef
        loss = policy_loss + (ent_coef * entropy_loss)
        
        # Log the metrics for monitoring training progress
        self.metrics_recorder.record_train({
            'loss' : loss.detach(),
            'policy_loss': policy_loss.detach(),
            'entropy_loss': entropy_loss.detach(),
            'entropy': entropy.detach(),
            'kl_div': kl_div.detach(),
            'approx_kl': approx_kl.detach(),
            'policy_targets_mean': policy_targets.mean().detach(),
            'policy_targets_std': policy_targets.std().detach()
        })
        return loss
