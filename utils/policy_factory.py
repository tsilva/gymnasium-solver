"""Policy factory utilities.

Centralizes creation of policy networks (MLP vs CNN) for both actor-critic and
policy-only variants. This encapsulates the logic that inspects observation
spaces to derive image shapes for CNN policies and forwards config kwargs.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch.nn as nn

from .models import (
    MLPActorCritic,
    CNNActorCritic,
    MLPPolicy,
    CNNPolicy,
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

        # Strong HWC signal: channels last with small channel count (e.g., 1,3,4)
        if W_last <= 8:
            return (int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2]))  # already HWC

        # CHW is used by our builder for images via VecTransposeImage and may be frame-stacked:
        # detect either small channel count (<=8) OR multiples of 3 (e.g., 12 for RGBx4)
        if (C_first <= 8 or (C_first % 3 == 0)) and (H_mid >= 16 and W_last >= 16):
            # Convert CHW -> HWC for downstream reshape utility
            return (H_mid, W_last, C_first)

        # Fallback: choose interpretation where last dim looks like channels
        if W_last <= 64:
            return (int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2]))
        # Otherwise assume CHW
        return (H_mid, W_last, C_first)
    if len(obs_shape) == 2:
        return (obs_shape[0], obs_shape[1], 1)
    # Fallback heuristic
    side = int(max(input_dim, 1) ** 0.5)
    return (side, side, 1)


def create_actor_critic_policy(
    policy_type: str,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    hidden_dims: Iterable[int],
    activation: str,
    obs_space=None,
    **policy_kwargs,
):
    if policy_type == 'mlp':
        return MLPActorCritic(
            input_shape, 
            hidden_dims, 
            output_shape, 
            activation,
            **policy_kwargs,
        )
    elif policy_type == 'cnn':
        hwc = _infer_hwc_from_space(obs_space, input_dim)
        return CNNActorCritic(
            input_shape, 
            hidden_dims, 
            output_shape, 
            activation,
            **policy_kwargs,
        )
    else:
        raise ValueError(f"Invalid policy type: {policy_type}")


def create_policy(
    policy_type: str | type[nn.Module],
    *,
    input_dim: int,
    action_dim: int,
    hidden_dims: Iterable[int],
    activation: str,
    obs_space=None,
    **policy_kwargs,
):
    if policy_type == 'mlp':  
        return MLPPolicy(
            input_dim, 
            action_dim, 
            hidden_dims=hidden_dims, 
            activation=activation,
            **policy_kwargs
        )
    elif policy_type == 'cnn':
        hwc = _infer_hwc_from_space(obs_space, input_dim)
        return CNNPolicy(
            obs_shape=hwc,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            **policy_kwargs,
        )
