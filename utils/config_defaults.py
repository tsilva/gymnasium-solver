"""Default resolution helpers for Config objects."""

from __future__ import annotations

import os
from dataclasses import MISSING

from utils.environment_types import get_env_type


def resolve_defaults(config) -> None:
    for field_info in config.__dataclass_fields__.values():
        value = getattr(config, field_info.name)
        if value is not None:
            continue
        if field_info.default is not MISSING:
            setattr(config, field_info.name, field_info.default)
        elif field_info.default_factory is not MISSING:
            setattr(config, field_info.name, field_info.default_factory())


def resolve_n_envs(config) -> None:
    if config.n_envs == "auto":
        config.n_envs = os.cpu_count() or 1


def resolve_policy(config) -> None:
    assert config.model_id is not None, (
        "model_id is required. Available models: mlp_tiny, mlp_small, mlp_medium, "
        "mlp_large, cnn_nature, cnn_impala, cnn_large"
    )

    from utils.model_registry import resolve_model_spec

    spec = resolve_model_spec(config.model_id)
    config.policy = spec.policy
    config._hidden_dims = spec.hidden_dims
    config._activation = spec.activation
    config._policy_kwargs = dict(spec.policy_kwargs)


def resolve_atari_defaults(config) -> None:
    if config.vectorization_mode not in ("atari", "auto"):
        return
    if get_env_type(config.env_id) != "alepy":
        return
    if config.obs_type != config.ObsType.rgb:
        return

    if config.grayscale_obs is None:
        config.grayscale_obs = True
    if config.resize_obs is None:
        config.resize_obs = (84, 84)
    if config.frame_stack is None:
        config.frame_stack = 4
    if config.frame_skip is None:
        config.frame_skip = 4


def resolve_vizdoom_defaults(config) -> None:
    if get_env_type(config.env_id) != "vizdoom":
        return

    if config.obs_type == config.ObsType.vector:
        config.obs_type = config.ObsType.rgb
    if config.grayscale_obs is None:
        config.grayscale_obs = True
    if config.resize_obs is None:
        config.resize_obs = (84, 84)
    if config.frame_stack is None:
        config.frame_stack = 4
    if config.model_id is None:
        config.model_id = "cnn_nature"
    if config.vectorization_mode in (None, "auto"):
        config.vectorization_mode = "async"


def resolve_retro_defaults(config) -> None:
    if get_env_type(config.env_id) != "stable_retro":
        return

    if config.obs_type == config.ObsType.vector:
        config.obs_type = config.ObsType.rgb
    if config.grayscale_obs is None:
        config.grayscale_obs = True
    if config.resize_obs is None:
        config.resize_obs = (84, 84)
    if config.frame_stack is None:
        config.frame_stack = 4
    if config.frame_skip is None:
        config.frame_skip = 4
    if config.vectorization_mode in (None, "auto"):
        config.vectorization_mode = "async"
    if config.model_id is None:
        config.model_id = "cnn_nature"


def resolve_numeric_strings(config) -> None:
    for key, value in list(vars(config).items()):
        if not isinstance(value, str):
            continue
        try:
            setattr(config, key, float(value))
        except (TypeError, ValueError):
            continue


def resolve_batch_size(config) -> None:
    if config.batch_size is None:
        policy_str = config.policy.value if hasattr(config.policy, "value") else str(config.policy)
        config.batch_size = 256 if "cnn" in policy_str else 64

    batch_size = config.batch_size
    if batch_size > 1:
        return

    rollout_size = config.n_envs * config.n_steps
    config.batch_size = max(1, int(rollout_size * batch_size))


def resolve_eval_warmup_epochs(config) -> None:
    warmup = config.eval_warmup_epochs
    if warmup <= 0 or warmup >= 1:
        return

    assert config.max_env_steps is not None, (
        "Fractional eval_warmup_epochs requires max_env_steps to be set"
    )

    total_epochs = config.max_env_steps / (config.n_envs * config.n_steps)
    config.eval_warmup_epochs = int(total_epochs * warmup)
