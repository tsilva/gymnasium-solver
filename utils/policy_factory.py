"""Policy factory utilities.

Centralizes creation of policy networks (MLP vs CNN) for both actor-critic and
policy-only variants. This encapsulates the logic that inspects observation
spaces to derive image shapes for CNN policies and forwards config kwargs.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch.nn as nn

from .models import (
    ActorCritic,
    CNNActorCritic,
    MLPPolicy,
    CNNPolicyOnly,
)


def _infer_hwc_from_space(obs_space, input_dim: int) -> Tuple[int, int, int]:
    """Infer an HWC observation shape from a Gymnasium observation space.

    Falls back to a square 1-channel guess based on input_dim if needed.
    """
    obs_shape = getattr(obs_space, "shape", None)
    if obs_shape is None:
        # Fallback heuristic
        side = int(max(input_dim, 1) ** 0.5)
        return (side, side, 1)
    if len(obs_shape) == 3:
        # Try to detect channel-first (C, H, W) vs channel-last (H, W, C)
        C_first, H_mid, W_last = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])
        # Heuristics: small channel count typically 1..8; spatial dims usually >= 16
        is_chw = (C_first <= 8 and H_mid >= 16 and W_last >= 16)
        if is_chw:
            # Convert CHW -> HWC for downstream reshape utility
            return (H_mid, W_last, C_first)
        # Otherwise assume HWC already
        return (C_first, H_mid, W_last)
    if len(obs_shape) == 2:
        return (obs_shape[0], obs_shape[1], 1)
    # Fallback heuristic
    side = int(max(input_dim, 1) ** 0.5)
    return (side, side, 1)


def create_actor_critic_policy(
    policy_type: str | type[nn.Module],
    *,
    input_dim: int,
    action_dim: int,
    hidden: Iterable[int] | int,
    activation: "str | type[nn.Module] | nn.Module" = "tanh",
    obs_space=None,
    **policy_kwargs,
):
    """Create an Actor-Critic policy model based on policy_type.

    policy_type: 'mlp' or 'cnn' (case-insensitive) or a Module class.
    activation: string or nn.Module class/instance; forwarded to underlying model.
    obs_space: Gymnasium observation space (required for CNN policies to infer shape).
    policy_kwargs: forwarded to the underlying model constructor.
    """
    # Accept direct module classes for extensibility
    if isinstance(policy_type, type) and issubclass(policy_type, nn.Module):
        return policy_type(input_dim, action_dim, hidden_dims=hidden, activation=activation, **policy_kwargs)

    if isinstance(policy_type, str) and policy_type.lower() in {"cnn", "cnnpolicy", "cnn_actor_critic", "cnnac"}:
        hwc = _infer_hwc_from_space(obs_space, input_dim)
        return CNNActorCritic(
            obs_shape=hwc,
            action_dim=action_dim,
            hidden=hidden,
            activation=activation,
            **policy_kwargs,
        )
    # Default: MLP-based actor-critic
    return ActorCritic(input_dim, action_dim, hidden_dims=hidden, activation=activation,)


def create_policy(
    policy_type: str | type[nn.Module],
    *,
    input_dim: int,
    action_dim: int,
    hidden_dims: Iterable[int] | int,
    activation: "str | type[nn.Module] | nn.Module" = "tanh",
    obs_space=None,
    **policy_kwargs,
):
    """Create a policy-only (no value head) model based on policy_type.

    Used by REINFORCE and other algorithms without a learned baseline.
    """
    if isinstance(policy_type, type) and issubclass(policy_type, nn.Module):
        return policy_type(input_dim, action_dim, hidden_dims=hidden_dims, activation=activation, **policy_kwargs)

    if isinstance(policy_type, str) and policy_type.lower() == "cnnpolicy":
        hwc = _infer_hwc_from_space(obs_space, input_dim)
        return CNNPolicyOnly(
            obs_shape=hwc,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            **policy_kwargs,
        )
    return MLPPolicy(
        input_dim, 
        action_dim, 
        hidden_dims=hidden_dims, 
        activation=activation
    )
