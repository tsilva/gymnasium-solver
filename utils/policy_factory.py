"""Policy factory utilities.

Centralizes creation of policy networks (MLP vs CNN) for both actor-critic and
policy-only variants. This encapsulates the logic that inspects observation
spaces to derive image shapes for CNN policies and forwards config kwargs.
"""

from __future__ import annotations

import torch.nn as nn

from .models import (
    MLPActorCritic,
    CNNActorCritic,
    MLPPolicy,
    CNNPolicy,
)

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
        "mlp_actorcritic": MLPActorCritic,
        "cnn_actorcritic": CNNActorCritic,
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
    if len(input_shape) == 1 and input_shape[0] == 1:
        input_shape = env.observation_space.high[0]

    output_shape = env.action_space.shape
    if not output_shape: output_shape = (env.action_space.n,)

    return build_policy(
        config.policy,
        input_shape=input_shape,
        output_shape=output_shape,
        hidden_dims=config.hidden_dims,
        activation=config.activation,
        **config.policy_kwargs,
    )
    