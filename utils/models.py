"""Neural network model utilities."""

import torch
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


class SharedBackboneNet(nn.Module):
    """Shared backbone network with separate policy and value heads for PPO."""
    
    def __init__(self, obs_dim, act_dim, hidden_dim=64, backbone_dim=None):
        super().__init__()
        
        # Use backbone_dim if specified, otherwise use hidden_dim
        if backbone_dim is None:
            backbone_dim = hidden_dim
            
        if isinstance(backbone_dim, (int, float)):
            backbone_dims = [int(backbone_dim)]
        else:
            backbone_dims = [int(dim) for dim in backbone_dim]
        
        # Shared backbone layers
        backbone_layers = []
        current_dim = obs_dim
        
        for hidden_size in backbone_dims:
            backbone_layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU()
            ])
            current_dim = hidden_size
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Separate heads for policy and value
        if isinstance(hidden_dim, (int, float)):
            head_dims = [int(hidden_dim)]
        else:
            head_dims = [int(dim) for dim in hidden_dim]
        
        # Policy head
        policy_layers = []
        policy_input_dim = current_dim
        for hidden_size in head_dims:
            policy_layers.extend([
                nn.Linear(policy_input_dim, hidden_size),
                nn.ReLU()
            ])
            policy_input_dim = hidden_size
        policy_layers.append(nn.Linear(policy_input_dim, act_dim))
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Value head
        value_layers = []
        value_input_dim = current_dim
        for hidden_size in head_dims:
            value_layers.extend([
                nn.Linear(value_input_dim, hidden_size),
                nn.ReLU()
            ])
            value_input_dim = hidden_size
        value_layers.append(nn.Linear(value_input_dim, 1))
        self.value_head = nn.Sequential(*value_layers)
    
    def forward(self, x):
        """Forward pass returning policy logits."""
        features = self.backbone(x)
        return self.policy_head(features)
    
    def get_value(self, x):
        """Get value estimates."""
        features = self.backbone(x)
        return self.value_head(features)
    
    def forward_both(self, x):
        """Forward pass returning both policy logits and values."""
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        values = self.value_head(features)
        return policy_logits, values


class SharedPolicyNet(nn.Module):
    """Wrapper around SharedBackboneNet to provide PolicyNet interface."""
    
    def __init__(self, shared_net):
        super().__init__()
        self.shared_net = shared_net
    
    def forward(self, x):
        return self.shared_net.forward(x)


class SharedValueNet(nn.Module):
    """Wrapper around SharedBackboneNet to provide ValueNet interface."""
    
    def __init__(self, shared_net):
        super().__init__()
        self.shared_net = shared_net
    
    def forward(self, x):
        return self.shared_net.get_value(x)
