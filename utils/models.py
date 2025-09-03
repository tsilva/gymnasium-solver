"""Neural network model utilities.

Includes simple MLP-based Actor-Critic and Policy-only models, plus CNN-based
variants for image observations (channels-last HWC expected from envs). CNN
models internally reshape flat inputs back to (N, C, H, W) using a provided
``obs_shape`` to remain compatible with the current rollout pipeline that
returns flattened observations.
"""

from typing import Iterable

import torch
from torch.distributions import Categorical
import torch.nn as nn
from .torch import init_model_weights

def resolve_activation(activation_id: str) -> type[nn.Module]:
    key = activation_id.lower()
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
    return mapping[key]


def build_mlp(input_shape: tuple[int, ...], hidden_dims: tuple[int, ...], activation:str):
    """ Create a stack of sequential linear layers with the given activation function. """
    assert len(input_shape) == 1, "Input shape must be 1D"
    activation_cls = resolve_activation(activation)
    layers = []
    last_dim = input_shape[0]
    for hidden_dim in hidden_dims:
        layers += [nn.Linear(last_dim, hidden_dim), activation_cls()]
        last_dim = hidden_dim
    return nn.Sequential(*layers)


class MLPPolicy(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: tuple[int, ...], 
        output_dim: int, 
        activation: str,
    ):
        super().__init__()

        # Normalize hidden dims and create MLP backbone
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.backbone = build_mlp(input_dim, hidden_dims, activation)

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

class MLPActorCritic(nn.Module):
    def __init__(
        self, 
        input_shape: tuple[int, ...], 
        hidden_dims: tuple[int, ...], 
        output_shape: tuple[int, ...], 
        activation: str
    ):
        super().__init__()

        assert len(input_shape) == 1, "Input shape must be 1D"
        assert len(output_shape) == 1, "Output shape must be 1D"

        self.backbone = build_mlp(input_shape, hidden_dims, activation)

        # Create the policy head
        self.policy_head = nn.Linear(hidden_dims[-1], output_shape[0])

        # Create the value head
        self.value_head = nn.Linear(hidden_dims[-1], 1)

        # TODO: review this
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
        policy_dist = Categorical(logits=logits)

        # Forward through value head and get value prediction
        value_pred = self.value_head(x).squeeze(-1)

        # Return policy distribution and value prediction
        return policy_dist, value_pred

    @torch.inference_mode()
    def act(self, obs: torch.Tensor, deterministic=False):
        policy_dist, v = self.forward(obs)
        a = policy_dist.sample() if not deterministic else policy_dist.mode
        return a, policy_dist.log_prob(a), v

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, v = self.forward(obs)
        return dist.log_prob(actions), dist.entropy(), v
    
    def predict_values(self, obs):
        x = self.backbone(obs)
        value = self.value_head(x).squeeze(-1)
        return value



class CNNActorCritic(nn.Module):
    """CNN-based Actor-Critic for discrete action spaces.

    Parameters are intentionally flexible; pass via config.policy_kwargs.
    """

    def __init__(
        self,
        obs_shape,  # expected HWC shape from env observation
        action_dim: int,
        hidden_dims: Iterable[int],
        activation: str,
        # CNN kwargs
        in_channels: int,
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
        self.backbone = build_mlp(feat_dim, hidden_dims, activation=activation) if hidden_dims and len(hidden_dims) > 0 else nn.Identity()
        last_dim = hidden_dims[-1] if hidden_dims and len(hidden_dims) > 0 else feat_dim
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
        # Normalize image inputs to [0,1] if they appear to be in 0..255 range
        # Avoid double-normalizing if inputs are already in [0,1].
        try:
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            else:
                # Convert to scalar to avoid tracing data-dependent control flow
                max_val = float(x.detach().amax().item())
                if max_val > 1.0:
                    x = x.div(255.0)
        except Exception:
            # Best-effort: fall through without normalization on errors
            pass
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


class CNNPolicy(nn.Module):
    """CNN-based policy-only network (no value head) for REINFORCE."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        hidden_dims: tuple[int, ...],
        output_shape: tuple[int, ...],
        activation: str,
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_sizes: tuple[int, ...] = (8, 4, 3),
        strides: tuple[int, ...] = (4, 2, 1),
    ):
        super().__init__()

        # Reshape flat inputs to images (N, C, H, W)
        self.reshape = _ReshapeFlatToImage(input_shape)
        H, W, C = self.reshape.hwc

        # Build a simple Conv2d stack directly from passed args (no NatureCNN)
        assert len(channels) == len(kernel_sizes) == len(strides), "Conv spec lengths must match"
        act_cls = resolve_activation(activation)

        conv_layers = []
        in_c = C
        for out_c, k, s in zip(channels, kernel_sizes, strides):
            conv_layers += [nn.Conv2d(in_c, out_c, kernel_size=k, stride=s), act_cls()]
            in_c = out_c
        self.convs = nn.Sequential(*conv_layers)

        # Determine feature size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, H, W, C)
            feat = self.convs(self.reshape(dummy))
            feat_dim = int(torch.flatten(feat, 1).shape[1])

        # Optional MLP head after CNN
        if hidden_dims and len(tuple(hidden_dims)) > 0:
            self.backbone = build_mlp((feat_dim,), tuple(hidden_dims), activation)
            last_dim = tuple(hidden_dims)[-1]
        else:
            self.backbone = nn.Identity()
            last_dim = feat_dim

        # Policy head
        self.policy_head = nn.Linear(last_dim, output_shape[0])

        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

    def _forward_features(self, obs_flat: torch.Tensor):
        x = self.reshape(obs_flat)
        # Normalize image inputs to [0,1] if likely uint8 scale
        try:
            if x.dtype == torch.uint8:
                x = x.float().div_(255.0)
            else:
                max_val = float(x.detach().amax().item())
                if max_val > 1.0:
                    x = x.div(255.0)
        except Exception:
            pass
        x = self.convs(x)
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

