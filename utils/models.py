"""Neural network model utilities."""

import torch.nn as nn


class MLPNet(nn.Module):
    """Reusable MLP with configurable hidden dimensions."""
    
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.ReLU):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_size in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                activation()
            ])
            current_dim = hidden_size
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class PolicyNet(MLPNet):
    """Policy network for RL agents."""
    
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__(input_dim, output_dim, hidden_dims)


class ValueNet(MLPNet):
    """Value network for RL agents."""
    
    def __init__(self, input_dim, hidden_dims):
        super().__init__(input_dim, 1, hidden_dims)


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
        self.pi = nn.Linear(hidden[-1], action_dim)
        self.v  = nn.Linear(hidden[-1], 1)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        dist = Categorical(logits=self.pi(x))
        value = self.v(x).squeeze(-1)
        return dist, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist, v = self.forward(obs)
        a = dist.sample()
        return a, dist.log_prob(a), v

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, v = self.forward(obs)
        return dist.log_prob(actions), dist.entropy(), v
        