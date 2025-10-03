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
    CNNActorCritic,
    MLPActorCritic,
    MLPPolicy,
)


def resolve_policy_type_for_config(config):
    """Auto-select CNN policy for configs with image observations.

    Updates config.policy in-place if auto-selection occurs.
    Returns the resolved policy type string.
    """
    policy_type = config.policy

    # Check if this is an environment with RGB observations
    is_atari = config.env_id.startswith("ALE/") or config.env_id.startswith("ALE-")
    is_retro = config.env_id.startswith("Retro/") or config.env_id.startswith("Retro-")
    is_vizdoom = config.env_id.startswith("VizDoom")
    is_rgb_obs = getattr(config, "obs_type", None) == "rgb"

    # Also check if render_mode is rgb_array which typically means image observations
    render_mode = getattr(config, "render_mode", None)
    has_rgb_render = render_mode == "rgb_array"

    is_image_config = (is_atari or is_retro or is_vizdoom) and (is_rgb_obs or has_rgb_render)

    if is_image_config:
        # Auto-upgrade MLP policies to CNN variants for image observations
        if policy_type == "mlp":
            policy_type = "cnn"
            config.policy = policy_type
        elif policy_type == "mlp_actorcritic":
            policy_type = "cnn_actorcritic"
            config.policy = policy_type

    return policy_type


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
    # Use single_observation_space for Gymnasium VectorEnv (excludes batch dimension)
    obs_space = getattr(env, 'single_observation_space', env.observation_space)
    input_shape = obs_space.shape
    if len(input_shape) == 1 and input_shape[0] == 1:
        input_shape = obs_space.high[0]

    # Use single_action_space for Gymnasium VectorEnv (excludes batch dimension)
    act_space = getattr(env, 'single_action_space', env.action_space)
    output_shape = act_space.shape
    if not output_shape: output_shape = (act_space.n,)

    # Auto-select CNN policy for multi-dimensional observations (images)
    policy_type = config.policy
    is_image_obs = len(input_shape) > 1
    if is_image_obs:
        if policy_type == "mlp":
            policy_type = "cnn"
        elif policy_type == "mlp_actorcritic":
            policy_type = "cnn_actorcritic"

    return build_policy(
        policy_type,
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
