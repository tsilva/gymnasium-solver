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
        input_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (64,),
        output_dim: int | None = None,
        activation: str = "relu",
        *,
        # Back-compat and factory-compat kwargs
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()

        # Harmonize input/output dims to support both direct ints and shape tuples
        if input_dim is None:
            assert input_shape is not None and len(input_shape) == 1, "Input shape must be 1D"
            input_dim = int(input_shape[0])
        if output_dim is None:
            assert output_shape is not None and len(output_shape) == 1, "Output shape must be 1D"
            output_dim = int(output_shape[0])

        # Normalize hidden dims and create MLP backbone
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.backbone = build_mlp((input_dim,), hidden_dims, activation)

        # Create the policy head
        self.policy_head = nn.Linear(hidden_dims[-1], output_dim)

        # Reusable initialization
        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        return dist, None  # Return None for value to maintain compatibility

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


class _ReshapeFlatToImage(nn.Module):
    """Utility module to reshape flat (N, prod(HWC)) to (N, C, H, W).

    Assumes original observation shape is HWC. Will transpose to CHW for CNNs.
    """

    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) >= 2, "obs_shape must be (H, W, C) or similar"
        if len(obs_shape) == 2:
            # No channels dimension; treat as single-channel
            H, W = int(obs_shape[0]), int(obs_shape[1])
            C = 1
            self.hwc = (H, W, C)
        elif len(obs_shape) >= 3:
            # Robustly resolve HWC from either HWC or CHW inputs.
            # Heuristics mirror utils.policy_factory._infer_hwc_from_space.
            a, b, c = int(obs_shape[-3]), int(obs_shape[-2]), int(obs_shape[-1])
            # If the last dim looks like channels (small count), assume HWC
            if c <= 8 and a >= 8 and b >= 8:
                H, W, C = a, b, c
            # If the first dim looks like channels (small or multiple of 3), assume CHW
            elif int(obs_shape[0]) <= 8 or (int(obs_shape[0]) % 3 == 0 and b >= 8 and c >= 8):
                H, W, C = int(obs_shape[1]), int(obs_shape[2]), int(obs_shape[0])
            # Fallbacks: pick the interpretation where spatial dims are larger
            elif c <= 64:
                H, W, C = a, b, c
            else:
                H, W, C = int(obs_shape[1]), int(obs_shape[2]), int(obs_shape[0])
            self.hwc = (H, W, C)
        else:
            # Extremely rare fallback; assume square single-channel
            side = int(max(int(obs_shape[0]), 1))
            self.hwc = (side, side, 1)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        H, W, C_expected = self.hwc
        # Flat inputs: (N, H*W*C)
        if x.ndim == 2:
            x = x.view(N, H, W, C_expected)
            return x.permute(0, 3, 1, 2).contiguous()
        # 4D inputs: allow HWC or CHW with arbitrary channel count (e.g., frame stack)
        if x.ndim == 4:
            # If channel-last (N, H, W, Cx), permute to channel-first
            if x.shape[1] == H and x.shape[2] == W:
                return x.permute(0, 3, 1, 2).contiguous()
            # If channel-first and spatial dims match, pass through
            if x.shape[2] == H and x.shape[3] == W:
                return x.contiguous()
        # Fallback: best-effort permute assuming input is HWC
        try:
            return x.permute(0, 3, 1, 2).contiguous()
        except Exception:
            return x


class _CNNTrunk(nn.Module):
    """Reusable CNN feature extractor with optional MLP head.

    Handles: reshape flat inputs to CHW, Conv2d stack, optional MLP, and
    image intensity normalization heuristics (uint8 → [0,1] or divide by 255).
    """

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        hidden_dims: Iterable[int] | None,
        activation: str,
        channels: Iterable[int] = (32, 64, 64),
        kernel_sizes: Iterable[int] = (8, 4, 3),
        strides: Iterable[int] = (4, 2, 1),
    ):
        super().__init__()

        # Reshape flat inputs to images (N, C, H, W)
        self.reshape = _ReshapeFlatToImage(input_shape)
        H, W, C = self.reshape.hwc

        # Conv stack
        assert len(tuple(channels)) == len(tuple(kernel_sizes)) == len(tuple(strides)), "Conv spec lengths must match"
        act_cls = resolve_activation(activation)
        conv_layers = []
        in_c = C
        for out_c, k, s in zip(channels, kernel_sizes, strides):
            conv_layers += [nn.Conv2d(in_c, out_c, kernel_size=int(k), stride=int(s)), act_cls()]
            in_c = out_c
        self.convs = nn.Sequential(*conv_layers)

        # Determine feature size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, H, W, C)
            feat = self.convs(self.reshape(dummy))
            feat_dim = int(torch.flatten(feat, 1).shape[1])

        # Optional MLP head after CNN
        hidden_dims_tuple = tuple(hidden_dims or ())
        if len(hidden_dims_tuple) > 0:
            self.backbone = build_mlp((feat_dim,), hidden_dims_tuple, activation)
            self.out_dim = hidden_dims_tuple[-1]
        else:
            self.backbone = nn.Identity()
            self.out_dim = feat_dim

    def forward_features(self, obs_flat: torch.Tensor) -> torch.Tensor:
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
        x = torch.flatten(x, 1)
        x = self.backbone(x)
        return x


class CNNActorCritic(nn.Module):
    """CNN-based Actor-Critic for discrete action spaces using a shared trunk."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int, int],
        hidden_dims: Iterable[int],
        output_shape: tuple[int, ...],
        activation: str,
        channels: Iterable[int] = (32, 64, 64),
        kernel_sizes: Iterable[int] = (8, 4, 3),
        strides: Iterable[int] = (4, 2, 1),
    ):
        super().__init__()

        self.trunk = _CNNTrunk(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            activation=activation,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
        )

        last_dim = self.trunk.out_dim
        self.policy_head = nn.Linear(last_dim, int(output_shape[0]))
        self.value_head = nn.Linear(last_dim, 1)

        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

    def forward(self, obs_flat: torch.Tensor):
        x = self.trunk.forward_features(obs_flat)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        value = self.value_head(x).squeeze(-1)
        return dist, value


class CNNPolicy(nn.Module):
    """CNN-based policy-only network (no value head) using the shared trunk."""

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

        self.trunk = _CNNTrunk(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            activation=activation,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
        )

        last_dim = self.trunk.out_dim
        self.policy_head = nn.Linear(last_dim, int(output_shape[0]))

        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

    def forward(self, obs_flat: torch.Tensor):
        x = self.trunk.forward_features(obs_flat)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        return dist, None
