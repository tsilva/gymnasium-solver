"""Neural network model utilities.

Includes simple MLP-based Actor-Critic and Policy-only models, plus CNN-based
variants for image observations (channels-last HWC expected from envs). CNN
models internally reshape flat inputs back to (N, C, H, W) using a provided
``obs_shape`` to remain compatible with the current rollout pipeline that
returns flattened observations.
"""

import math
from typing import Iterable

import torch
from torch.distributions import Categorical
import torch.nn as nn
from .torch import init_model_weights


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
    # Allow an int or an iterable of ints
    if isinstance(hidden, int):
        hidden = (hidden,)
    act_cls = resolve_activation(act)
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act_cls()]
        last = h
    return nn.Sequential(*layers)


class NatureCNN(nn.Module):
    """A lightweight, configurable CNN feature extractor.

    Expects input tensors as (N, C, H, W). If observations arrive flattened
    (N, H*W*C), callers should reshape prior to calling forward or pass through
    a wrapper model that reshapes.

    Default mirrors common RL baselines for 84x84 inputs, but parameters are
    configurable via kwargs.
    """

    def __init__(
        self,
        in_channels: int,
        channels=(32, 64, 64),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1),
        activation: "str | type[nn.Module] | nn.Module" = nn.ReLU,
        flatten: bool = True,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(strides), "Conv spec lengths must match"
        act_cls = resolve_activation(activation)

        convs = []
        last_c = in_channels
        for c, k, s in zip(channels, kernel_sizes, strides):
            convs += [nn.Conv2d(last_c, c, kernel_size=k, stride=s), act_cls()]
            last_c = c
        self.convs = nn.Sequential(*convs)
        self.flatten = flatten

    def forward(self, x):
        x = self.convs(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        return x

class MLPPolicy(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: Iterable[int] | int = (64, 64), activation: "str | type[nn.Module] | nn.Module" = nn.Tanh
    ):
        super().__init__()

        # Normalize hidden dims and create MLP backbone
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.backbone = mlp(input_dim, hidden_dims, activation)

        # Create the policy head
        self.policy_head = nn.Linear(hidden_dims[-1], output_dim)

        # Reusable initialization
        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

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
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int] | int = (64, 64), activation: "str | type[nn.Module] | nn.Module" = nn.Tanh):
        super().__init__()

        # Normalize hidden dims and create MLP backbone
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.backbone = mlp(input_dim, hidden_dims, activation)

        # Create the policy head
        self.policy_head = nn.Linear(hidden_dims[-1], output_dim)

        # Create the value head
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        # Reusable initialization
        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

    def forward(self, obs: torch.Tensor):
        # Forward observation through backbone
        x = self.backbone(obs)

        # Forward through policy head and get policy logits
        logits = self.policy_head(x)

        # Create categorical distribution from logits
        dist = Categorical(logits=logits)

        # Forward through value head and get value
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


class _ReshapeFlatToImage(nn.Module):
    """Utility module to reshape flat (N, prod(HWC)) to (N, C, H, W).

    Assumes original observation shape is HWC. Will transpose to CHW for CNNs.
    """

    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) >= 2, "obs_shape must be (H, W, C) or similar"
        if len(obs_shape) == 2:
            # No channels dimension; treat as single-channel
            H, W = obs_shape
            C = 1
            self.hwc = (H, W, C)
        else:
            H, W, C = obs_shape[-3:]
            self.hwc = (H, W, C)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        H, W, C = self.hwc
        if x.ndim == 2:
            x = x.view(N, H, W, C)
        # Convert HWC -> CHW
        x = x.permute(0, 3, 1, 2).contiguous()
        # Normalize uint8-like ranges if needed (assume already float)
        return x


class CNNActorCritic(nn.Module):
    """CNN-based Actor-Critic for discrete action spaces.

    Parameters are intentionally flexible; pass via config.policy_kwargs.
    """

    def __init__(
        self,
        obs_shape,  # expected HWC shape from env observation
        action_dim: int,
        hidden=(256,),
        activation: "str | type[nn.Module] | nn.Module" = nn.ReLU,
        # CNN kwargs
        in_channels: int | None = None,
        channels=(32, 64, 64),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1),
    ):
        super().__init__()
        self.reshape = _ReshapeFlatToImage(obs_shape)
        H, W, C = self.reshape.hwc
        C_in = in_channels if in_channels is not None else C
        act_cls = resolve_activation(activation)

        # CNN feature extractor
        self.cnn = NatureCNN(C_in, channels=channels, kernel_sizes=kernel_sizes, strides=strides, activation=activation)

        # To build the MLP head dimensions, run a dummy forward with zeros
        with torch.no_grad():
            dummy = torch.zeros(1, H, W, C)
            feat = self.cnn(self.reshape(dummy))
            feat_dim = feat.shape[1]

        # MLP head shared by policy and value
        self.backbone = mlp(feat_dim, hidden, act=activation) if hidden and len(hidden) > 0 else nn.Identity()
        last_dim = hidden[-1] if hidden and len(hidden) > 0 else feat_dim
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)

        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

    def _forward_features(self, obs_flat: torch.Tensor):
        x = self.reshape(obs_flat)
        x = self.cnn(x)
        x = self.backbone(x)
        return x

    def forward(self, obs_flat: torch.Tensor):
        x = self._forward_features(obs_flat)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        value = self.value_head(x).squeeze(-1)
        return dist, value

    @torch.inference_mode()
    def act(self, obs_flat: torch.Tensor, deterministic=False):
        dist, v = self.forward(obs_flat)
        a = dist.sample() if not deterministic else dist.mode
        return a, dist.log_prob(a), v

    def evaluate_actions(self, obs_flat: torch.Tensor, actions: torch.Tensor):
        dist, v = self.forward(obs_flat)
        return dist.log_prob(actions), dist.entropy(), v

    def predict_values(self, obs_flat: torch.Tensor):
        x = self._forward_features(obs_flat)
        return self.value_head(x).squeeze(-1)


class CNNPolicyOnly(nn.Module):
    """CNN-based policy-only network (no value head) for REINFORCE."""

    def __init__(
        self,
        obs_shape,
        action_dim: int,
        hidden_dims=(256,),
        activation: "str | type[nn.Module] | nn.Module" = nn.ReLU,
        in_channels: int | None = None,
        channels=(32, 64, 64),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1),
    ):
        super().__init__()
        self.reshape = _ReshapeFlatToImage(obs_shape)
        H, W, C = self.reshape.hwc
        C_in = in_channels if in_channels is not None else C

        self.cnn = NatureCNN(C_in, channels=channels, kernel_sizes=kernel_sizes, strides=strides, activation=activation)
        with torch.no_grad():
            feat = self.cnn(self.reshape(torch.zeros(1, H, W, C)))
            feat_dim = feat.shape[1]
        self.backbone = mlp(feat_dim, hidden_dims, act=activation) if hidden_dims and len(hidden_dims) > 0 else nn.Identity()
        last_dim = hidden_dims[-1] if hidden_dims and len(hidden_dims) > 0 else feat_dim
        self.policy_head = nn.Linear(last_dim, action_dim)

        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

    def _forward_features(self, obs_flat: torch.Tensor):
        x = self.reshape(obs_flat)
        x = self.cnn(x)
        x = self.backbone(x)
        return x

    def forward(self, obs_flat: torch.Tensor):
        x = self._forward_features(obs_flat)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        return dist, None

    @torch.inference_mode()
    def act(self, obs_flat: torch.Tensor, deterministic=False):
        dist, _ = self.forward(obs_flat)
        a = dist.sample() if not deterministic else dist.mode
        return a, dist.log_prob(a), torch.zeros_like(a, dtype=torch.float32)

    def evaluate_actions(self, obs_flat: torch.Tensor, actions: torch.Tensor):
        dist, _ = self.forward(obs_flat)
        return dist.log_prob(actions), dist.entropy(), torch.zeros(obs_flat.shape[0], device=obs_flat.device)

    def predict_values(self, obs_flat: torch.Tensor):
        return torch.zeros(obs_flat.shape[0], device=obs_flat.device)

