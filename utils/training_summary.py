"""Pre-fit training summary builder.

Constructs and displays environment, training, and run configuration
summaries before training begins.
"""

from __future__ import annotations

from utils.logging import display_config_summary


def _format_summary_value(value):
    if isinstance(value, (dict, list, tuple, set)):
        return str(value)
    return value


def present_prefit_summary(config) -> None:
    """Display a pre-fit summary of environment, training, and run configuration."""
    spec = config.spec if isinstance(getattr(config, "spec", None), dict) else {}
    returns = spec.get("returns") if isinstance(spec, dict) else {}
    rewards = spec.get("rewards") if isinstance(spec, dict) else {}

    reward_threshold = None
    if isinstance(returns, dict) and returns.get("threshold_solved") is not None:
        reward_threshold = returns.get("threshold_solved")
    elif isinstance(rewards, dict) and rewards.get("threshold_solved") is not None:
        reward_threshold = rewards.get("threshold_solved")

    env_block = {
        "env_id": _format_summary_value(config.env_id),
        "obs_type": _format_summary_value(config.obs_type),
        "wrappers": _format_summary_value(config.env_wrappers),
        "n_envs": _format_summary_value(getattr(config, "n_envs", None)),
        "vectorization_mode": _format_summary_value(config.vectorization_mode),
        "frame_stack": _format_summary_value(getattr(config, "frame_stack", None)),
        "normalize_obs": _format_summary_value(getattr(config, "normalize_obs", None)),
        "normalize_reward": _format_summary_value(getattr(config, "normalize_reward", None)),
        "grayscale_obs": _format_summary_value(getattr(config, "grayscale_obs", None)),
        "resize_obs": _format_summary_value(getattr(config, "resize_obs", None)),
        "spec/action_space": _format_summary_value(spec.get("action_space")),
        "spec/observation_space": _format_summary_value(spec.get("observation_space")),
        "reward_threshold": _format_summary_value(reward_threshold),
        "time_limit": _format_summary_value(spec.get("max_episode_steps")),
    }

    training_block = {
        "algo_id": _format_summary_value(getattr(config, "algo_id", None)),
        "policy": _format_summary_value(getattr(config, "policy", None)),
        "hidden_dims": _format_summary_value(getattr(config, "hidden_dims", None)),
        "activation": _format_summary_value(getattr(config, "activation", None)),
        "optimizer": _format_summary_value(getattr(config, "optimizer", None)),
        "accelerator": _format_summary_value(getattr(config, "accelerator", None)),
        "seed": _format_summary_value(getattr(config, "seed", None)),
        "n_envs": _format_summary_value(getattr(config, "n_envs", None)),
        "n_steps": _format_summary_value(getattr(config, "n_steps", None)),
        "n_epochs": _format_summary_value(getattr(config, "n_epochs", None)),
        "batch_size": _format_summary_value(getattr(config, "batch_size", None)),
        "max_env_steps": _format_summary_value(getattr(config, "max_env_steps", None)),
        "policy_lr": _format_summary_value(getattr(config, "policy_lr", None)),
        "gamma": _format_summary_value(getattr(config, "gamma", None)),
        "gae_lambda": _format_summary_value(getattr(config, "gae_lambda", None)),
        "ent_coef": _format_summary_value(getattr(config, "ent_coef", None)),
        "vf_coef": _format_summary_value(getattr(config, "vf_coef", None)),
        "clip_range": _format_summary_value(getattr(config, "clip_range", None)),
        "max_grad_norm": _format_summary_value(getattr(config, "max_grad_norm", None)),
        "returns_type": _format_summary_value(getattr(config, "returns_type", None)),
        "normalize_returns": _format_summary_value(getattr(config, "normalize_returns", None)),
        "advantages_type": _format_summary_value(getattr(config, "advantages_type", None)),
        "normalize_advantages": _format_summary_value(getattr(config, "normalize_advantages", None)),
        "target_kl": _format_summary_value(getattr(config, "target_kl", None)),
    }

    project_id = getattr(config, "project_id", None) or getattr(config, "env_id", None)
    run_block = {
        "project_id": _format_summary_value(project_id),
        "run_id": "<pending>",
        "quiet": _format_summary_value(getattr(config, "quiet", False)),
    }

    display_config_summary({
        "Run": run_block,
        "Environment": env_block,
        "Training": training_block,
    })
