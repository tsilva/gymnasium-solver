import torch

from utils.torch import assert_detached, batch_normalize, compute_kl_metrics, normalize_batch_with_metrics

from ..base_agent import BaseAgent


class REINFORCEAgent(BaseAgent):

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
        returns, returns_norm_metrics = normalize_batch_with_metrics(
            returns, self.config.normalize_returns, "roll/return"
        )

        # Normalize advantages if requested
        advantages, adv_norm_metrics = self._normalize_advantages(advantages)

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

        # The final loss is the sum of the policy loss and the entropy loss;
        # the higher the entropy coefficient the more priority we give to exploration
        ent_coef = self.ent_coef
        loss = policy_loss + (ent_coef * entropy_loss)

        # Compute KL metrics (matches PPO-style on-action KL estimates)
        kl_metrics, _, _ = compute_kl_metrics(old_logprobs, logprobs)

        # Log the metrics for monitoring training progress
        common_metrics = self._build_common_metrics(loss, policy_loss, entropy_loss, entropy)
        metrics = {
            **common_metrics,
            'policy_targets_mean': policy_targets.mean().detach(),
            'policy_targets_std': policy_targets.std().detach(),
            **kl_metrics,
            **returns_norm_metrics,
            **adv_norm_metrics,
        }

        self.metrics_recorder.record("train", metrics)

        # Return result for training step
        return dict(
            loss=loss,
            early_stop_epoch=False,
        )
