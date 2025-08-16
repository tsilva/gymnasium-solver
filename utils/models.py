"""Neural network model utilities."""

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical


def resolve_activation(act: "str | type[nn.Module] | nn.Module" = nn.Tanh) -> type[nn.Module]:
    """Map a string or nn.Module class/instance to an activation module class.

    Supported strings: 'tanh','relu','leaky_relu','elu','selu','gelu','silu','swish','identity'.
    Defaults to nn.Tanh.
    """
    if isinstance(act, type) and issubclass(act, nn.Module):
        return act
    if isinstance(act, nn.Module):
        return act.__class__
    if isinstance(act, str):
        key = act.lower()
        mapping = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "swish": nn.SiLU,  # alias
            "identity": nn.Identity,
        }
        return mapping.get(key, nn.Tanh)
    return nn.Tanh


def mlp(in_dim, hidden, act: "str | type[nn.Module] | nn.Module" = nn.Tanh):
    act_cls = resolve_activation(act)
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act_cls()]
        last = h
    return nn.Sequential(*layers)

class PolicyOnly(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(64, 64), activation: "str | type[nn.Module] | nn.Module" = nn.Tanh):
        super().__init__()
        self.backbone = mlp(state_dim, hidden, activation)
        self.policy_head = nn.Linear(hidden[-1], action_dim)
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for backbone
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small init for policy head
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        return dist, None  # Return None for value to maintain compatibility

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic=False):
        dist, _ = self.forward(obs)
        a = dist.sample() if not deterministic else dist.mode
        return a, dist.log_prob(a), torch.zeros_like(a, dtype=torch.float32)  # Return zero values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, _ = self.forward(obs)
        return dist.log_prob(actions), dist.entropy(), torch.zeros(obs.shape[0], device=obs.device)
    
    def predict_values(self, obs):
        # Return zeros for compatibility - REINFORCE doesn't use value function
        return torch.zeros(obs.shape[0], device=obs.device)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(64, 64), activation: "str | type[nn.Module] | nn.Module" = nn.Tanh):
        super().__init__()
        self.backbone = mlp(state_dim, hidden, activation)
        self.policy_head = nn.Linear(hidden[-1], action_dim)
        self.value_head  = nn.Linear(hidden[-1], 1)
        self._init_weights()

    def _init_weights(self):
        # Orthogonal init for backbone
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small init for policy head helps PPO stability
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        # Value head with unit gain
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        value = self.value_head(x).squeeze(-1)
        return dist, value

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic=False):
        dist, v = self.forward(obs)
        a = dist.sample() if not deterministic else dist.mode
        return a, dist.log_prob(a), v

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, v = self.forward(obs)
        return dist.log_prob(actions), dist.entropy(), v
    
    def predict_values(self, obs):
        x = self.backbone(obs)
        value = self.value_head(x).squeeze(-1)
        return value

