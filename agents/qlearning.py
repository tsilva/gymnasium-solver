import torch
import random
from .base_agent import BaseAgent


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
    
    def train_on_batch(self, batch, batch_idx):
        states = batch.observations.long()   # Ensure they're usable as indices
        actions = batch.actions.long()
        returns = batch.returns

        # Gather current Q values
        current_q = self.q_table[states, actions]

        # Calculate target Q values
        max_next_q = torch.max(self.q_table[states], dim=1).values
        target_q = returns + 0.99 * max_next_q

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