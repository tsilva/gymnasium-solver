"""Policy factory utilities.

Centralizes creation of policy networks (MLP vs CNN) for both actor-critic and
policy-only variants. This encapsulates the logic that inspects observation
spaces to derive image shapes for CNN policies and forwards config kwargs.
"""

from __future__ import annotations

from typing import Iterable, Union

import torch.nn as nn

from .models import (
    MLPActorCritic,
    CNNActorCritic,
    MLPPolicy,
    CNNPolicy,
)

def build_actor_critic_policy(
    policy_type: str,
    *,
    input_shape: Union[tuple[int, ...], int],
    output_shape: tuple[int, ...],
    hidden_dims: Iterable[int],
    activation: str,
    **policy_kwargs,
):
    if policy_type == 'mlp':
        return MLPActorCritic(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            output_shape=output_shape,
            activation=activation,
            **policy_kwargs,
        )
    elif policy_type == 'cnn':
        return CNNActorCritic(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            output_shape=output_shape,
            activation=activation,
            **policy_kwargs,
        )
    else:
        raise ValueError(f"Invalid policy type: {policy_type}")


def build_policy(
    policy_type: str | type[nn.Module],
    *,
    input_shape: tuple[int, ...],
    hidden_dims: tuple[int, ...],
    output_shape: tuple[int, ...],
    activation: str,
    **policy_kwargs,
):
    policy_cls = {
        "mlp": MLPPolicy,
        "cnn": CNNPolicy,
    }[policy_type]
    
    policy = policy_cls(
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        output_shape=output_shape,
        activation=activation,
        **policy_kwargs,
    )
    return policy

def build_policy_from_env_and_config(env, config):
    # TODO: hack to force embeddings
    input_shape = env.observation_space.shape
    if len(input_shape) == 1:
        input_shape = env.observation_space.high[0]

    output_shape = env.action_space.shape
    if not output_shape: output_shape = (env.action_space.n,)
    policy_type = config.policy
    activation = config.activation
    policy_kwargs = config.policy_kwargs
    return build_actor_critic_policy(
        policy_type,
        input_shape=input_shape,
        output_shape=output_shape,
        hidden_dims=config.hidden_dims,
        activation=activation,
        # TODO: redundancy with input_dim/output_dim?
        **policy_kwargs,
    )
    