import torch

from utils.policy_factory import build_policy_from_env_and_config
from utils.torch import assert_detached, batch_normalize, compute_kl_diagnostics

from ..base_agent import BaseAgent


class REINFORCEAgent(BaseAgent):

    def build_models(self):
        train_env = self.get_env("train")
        self.policy_model = build_policy_from_env_and_config(train_env, self.config)

    # TODO: only does something with normalization off, but even that way it doesnt converge
    def losses_for_batch(self, batch, batch_idx):
        # Retrieve tensors from batch
        observations = batch.observations
        actions = batch.actions
        old_logprobs = batch.logprobs
        returns = batch.returns
        advantages = batch.advantages

        # Assert that the tensors are detached
        assert_detached(observations, actions, returns, advantages)
        
        # Normalize returns if requested
        if self.config.normalize_returns == "batch":
            returns = batch_normalize(returns)

        # Normalize advantages if requested
        if self.config.normalize_advantages == "batch":
            advantages = batch_normalize(advantages)

        # Pick the configured policy targets
        if self.config.policy_targets == "returns": 
            policy_targets = returns
        elif self.config.policy_targets == "advantages": 
            policy_targets = advantages
        else: 
            raise ValueError(f"Invalid policy targets: {self.config.policy_targets}")

        # Get the log probabilities for each
        # action given the current observation
        dist, _ = self.policy_model(observations)
        logprobs = dist.log_prob(actions)

        # Scale each action log probability by the monte carlo return for that observation;
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
            kl_div, approx_kl = compute_kl_diagnostics(old_logprobs, logprobs)
        
        # The final loss is the sum of the policy loss and the entropy loss;
        # the higher the entropy coefficient the more priority we give to exploration
        ent_coef = self.ent_coef
        loss = policy_loss + (ent_coef * entropy_loss)
        
        # Log the metrics for monitoring training progress
        self.metrics_recorder.record("train", {
            'opt/loss/total' : loss.detach(),
            'opt/loss/policy': policy_loss.detach(),
            'opt/loss/entropy': entropy_loss.detach(),
            'opt/policy/entropy': entropy.detach(),
            'opt/ppo/kl': kl_div.detach(),
            'opt/ppo/approx_kl': approx_kl.detach(),
            'policy_targets_mean': policy_targets.mean().detach(),
            'policy_targets_std': policy_targets.std().detach()
        })

        # Return result for training step
        return dict(
            loss=loss,
            early_stop_epoch=False,
        )
