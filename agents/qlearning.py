import random

import torch

from .base_agent import BaseAgent


# TODO: should we get rid of models.py and move models next to agents?
class QLearningPolicyModel():
    def __init__(self, q_table, action_space):
        self.q_table = q_table
        self.action_space = action_space
        self.exploration_rate = 0.99
        self.exploration_rate_decay = 0.99999

    def act(self, obs, deterministic=False):
        if not deterministic and random.random() < self.exploration_rate: 
            actions = torch.tensor([self.action_space.sample() for _ in range(obs.shape[0])], device=obs.device)
        else: 
            available_actions = self.q_table[obs.int()]
            actions = torch.argmax(available_actions, dim=1)
        logps_t = torch.zeros(obs.shape[0], device=obs.device)
        values_t = torch.zeros(obs.shape[0], device=obs.device)
        return actions, logps_t, values_t

    def predict_values(self, obs):
        return torch.zeros(obs.shape[0], device=obs.device)

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_rate_decay

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