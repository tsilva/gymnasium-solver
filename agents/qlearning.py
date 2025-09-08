import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict

from .base_agent import BaseAgent
from utils.torch import compute_param_group_grad_norm


class QLearningPolicyModel(nn.Module):
    """Forward-only Q-learning policy with epsilon-greedy behavior.

    forward(obs) returns a Categorical distribution whose probabilities are a
    blend of a one-hot greedy action and a uniform distribution based on the
    current exploration rate (epsilon). No value head is returned (None).
    """

    def __init__(self, q_table: torch.Tensor, action_space):
        super().__init__()
        # Keep reference to external Q-table tensor (updated by agent)
        self.q_table = q_table
        self.action_space = action_space
        self.exploration_rate = 0.99
        self.exploration_rate_decay = 0.99999
        # Dummy buffer to expose a device for utilities
        self.register_buffer("_dev", torch.zeros(1), persistent=False)

    def forward(self, obs: torch.Tensor):
        # obs expected as integer state indices (possibly batched)
        q_vals = self.q_table[obs.long()]  # shape: (B, n_actions)
        # Greedy action per sample
        greedy = torch.argmax(q_vals, dim=1)
        n_actions = int(q_vals.shape[1])

        # Build epsilon-greedy probabilities: (1-eps) on greedy, eps uniform elsewhere
        eps = float(self.exploration_rate)
        probs = torch.full_like(q_vals, fill_value=eps / max(n_actions, 1), dtype=torch.float32)
        probs.scatter_(1, greedy.view(-1, 1), (1.0 - eps) + eps / max(n_actions, 1))
        dist = Categorical(probs=probs)
        return dist, None

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_rate_decay

    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for Q-learning policy.
        
        Q-learning doesn't use gradients in the traditional sense since it updates
        the Q-table directly, but we provide this for compatibility.
        """
        # Q-learning doesn't have trainable parameters in the traditional sense
        # The Q-table is updated directly, not through gradients
        return {
            "grad_norm/q_table": 0.0  # No gradients for Q-table updates
        }

class QLearning(BaseAgent):
    
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False

    def create_models(self):
        self.exploration_rate = 0.99
        self.exploration_rate_decay = 0.99999
        self.q_table = torch.zeros((self.train_env.observation_space.n, self.train_env.action_space.n))
        self.policy_model = QLearningPolicyModel(self.q_table, self.train_env.action_space)
    
    # TODO: handle truncattions as well
    def losses_for_batch(self, batch, batch_idx):
        states = batch.observations.long()   # Ensure they're usable as indices
        actions = batch.actions.long()
        rewards = batch.rewards
        dones = batch.dones
        next_states = batch.next_observations.long()  # Use actual next states

        # Gather current Q values
        states = states.squeeze(-1) 
        current_q = self.q_table[states, actions]
        
        # Calculate max Q-values for next states
        next_states = next_states.squeeze(-1)
        next_q = self.q_table[next_states]
        max_next_q = torch.max(next_q, dim=1).values
        
        # Calculate target Q values using Bellman equation
        # target_q = reward + gamma * max_q(next_state) * (1 - done)
        gamma = 0.99 # TODO: softcode this
        target_q = rewards + gamma * max_next_q * (~dones).float()

        # Compute loss (mean squared error)
        loss = torch.mean((current_q - target_q.detach()) ** 2)

        # Q-table update (in-place)
        self.q_table[states, actions] = target_q.detach()

        self.log_metrics({
            'loss': loss.item(),
        }, prefix="train")
        return loss
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.policy_model.decay_exploration()

    def configure_optimizers(self):
        return []
