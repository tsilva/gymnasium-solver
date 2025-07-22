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

