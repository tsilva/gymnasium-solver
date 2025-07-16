"""Neural network model utilities."""

import torch.nn as nn
from typing import Union, Tuple


class MLPNet(nn.Module):
    """Reusable MLP with configurable hidden dimensions."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=64, activation=nn.ReLU):
        super().__init__()
        
        if isinstance(hidden_dim, (int, float)):
            hidden_dims = [int(hidden_dim)]
        else:
            hidden_dims = [int(dim) for dim in hidden_dim]
        
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
    
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__(obs_dim, act_dim, hidden_dim)


class ValueNet(MLPNet):
    """Value network for RL agents."""
    
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__(obs_dim, 1, hidden_dim)
