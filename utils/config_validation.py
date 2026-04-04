"""Validation helpers for Config objects."""

from __future__ import annotations

from utils.environment_types import get_env_type


def validate_common_config(config, *, config_enum_cls, logger) -> None:
    config._validate_positive("seed", allow_none=False)
    config._validate_positive("n_envs", allow_none=False)
    config._validate_positive("policy_lr")
    config._validate_non_negative("ent_coef")
    config._validate_positive("n_epochs")
    config._validate_positive("n_steps")
    config._validate_positive("batch_size")
    config._validate_positive("max_env_steps")
    config._validate_positive("frame_skip")
    config._validate_range("gamma", 0, 1)
    config._validate_positive("eval_freq_epochs")
    config._validate_non_negative("eval_warmup_epochs", allow_none=False)
    config._validate_positive("eval_episodes")
    config._validate_positive("reward_threshold")

    if config.max_env_steps is not None and config.max_env_steps % config.n_envs != 0:
        rounded = round(config.max_env_steps / config.n_envs) * config.n_envs
        logger.warning(
            f"max_env_steps ({config.max_env_steps}) not divisible by n_envs ({config.n_envs}). "
            f"Auto-rounding to {rounded}."
        )
        config.max_env_steps = rounded

    if config.devices is not None and not (isinstance(config.devices, int) or config.devices == "auto"):
        raise ValueError("devices may be an int, 'auto', or None.")

    if config.vectorization_mode not in {"auto", "atari", "sync", "async", None}:
        raise ValueError(
            "vectorization_mode must be 'auto', 'atari', 'sync', 'async', or None, "
            f"got: {config.vectorization_mode}"
        )

    if config.vectorization_mode == "atari":
        if get_env_type(config.env_id) != "alepy":
            raise ValueError(
                "vectorization_mode='atari' is only valid for Atari environments (ALE/*), "
                f"got env_id: {config.env_id}"
            )
        if config.obs_type != config_enum_cls.ObsType.rgb:
            raise ValueError(
                "vectorization_mode='atari' is only valid for RGB observations, "
                f"got obs_type: {config.obs_type}"
            )

    if config.n_envs is not None and config.n_steps is not None and config.batch_size is not None:
        rollout_size = config.n_envs * config.n_steps
        if config.batch_size > rollout_size:
            raise ValueError(
                f"batch_size ({config.batch_size}) should not exceed "
                f"n_envs ({config.n_envs}) * n_steps ({config.n_steps})."
            )
        if rollout_size % int(config.batch_size) != 0:
            raise ValueError(
                "batch_size must divide (n_envs * n_steps) exactly to yield uniform minibatches: "
                f"rollout_size={rollout_size}, batch_size={config.batch_size}."
            )

    if config.policy_targets is not None and config.policy_targets not in {
        config_enum_cls.PolicyTargetsType.returns,
        config_enum_cls.PolicyTargetsType.advantages,
    }:
        raise ValueError("policy_targets must be 'returns' or 'advantages'.")

    config._validate_non_negative("replay_ratio")
    config._validate_non_negative("replay_buffer_size")
    config._validate_positive("replay_is_clip")
    config._validate_schedules()


def validate_ppo_config(config, *, config_enum_cls) -> None:
    config._validate_positive("target_kl")
    config._validate_range("gae_lambda", 0, 1)
    config._validate_range("clip_range", 0, 1, inclusive_min=False, inclusive_max=False)
    config._validate_range("clip_range_vf", 0, 1, inclusive_min=False, inclusive_max=False)
    config._validate_non_negative("vf_coef")

    if config.normalize_advantages is not None and config.normalize_advantages not in {
        config_enum_cls.AdvantageNormType.rollout,
        config_enum_cls.AdvantageNormType.batch,
        config_enum_cls.AdvantageNormType.off,
    }:
        raise ValueError("normalize_advantages must be 'rollout', 'batch', or 'off'.")
