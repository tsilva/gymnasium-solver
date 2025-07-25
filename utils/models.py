"""Neural network model utilities."""

import torch
import torch.nn as nn
from torch.distributions import Categorical

def mlp(in_dim, hidden, act=nn.Tanh):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(64, 64)):
        super().__init__()
        self.backbone = mlp(state_dim, hidden, nn.Tanh)
        self.policy_head = nn.Linear(hidden[-1], action_dim)
        self.value_head  = nn.Linear(hidden[-1], 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        dist = Categorical(logits=self.policy_head(x))
        value = self.value_head(x).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist, v = self.forward(obs)
        a = dist.sample()
        return a, dist.log_prob(a), v

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, v = self.forward(obs)
        return dist.log_prob(actions), dist.entropy(), v
        