"""Policy factory utilities.

Centralizes creation of policy networks for both actor-critic and policy-only
MLP variants. For image observations, higher-level code is expected to provide
compatible shapes; this module focuses on model instantiation and wiring.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .models import (
    MLPActorCritic,
    MLPPolicy,
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
        "mlp_actorcritic": MLPActorCritic,
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


def load_policy_model_from_checkpoint(
    ckpt_path: Path,
    env,
    config,
):
    """Instantiate a policy for the given env/config and load weights from checkpoint."""

    policy_model = build_policy_from_env_and_config(env, config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = None
    if isinstance(ckpt, dict):
        maybe = ckpt.get("model_state_dict")
        if isinstance(maybe, dict):
            state_dict = maybe
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt  # fallback: treat whole dict as state dict
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Invalid checkpoint: missing model_state_dict in {ckpt_path}")

    policy_model.load_state_dict(state_dict)
    policy_model.eval()
    return policy_model, ckpt
