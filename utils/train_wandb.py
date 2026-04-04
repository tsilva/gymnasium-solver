"""Weights & Biases lifecycle helpers for training flows."""

from __future__ import annotations

import os
from dataclasses import asdict

import wandb

from utils.config import Config
from utils.formatting import sanitize_name


def maybe_merge_wandb_config(config, *, wandb_sweep_flag: bool):
    """Merge active W&B sweep overrides back into the Config instance."""
    wandb_sweep_id = os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    is_wandb_sweep = bool(wandb_sweep_flag) or bool(wandb_sweep_id)
    if not is_wandb_sweep:
        return config

    project_name = config.project_id if config.project_id else sanitize_name(config.env_id)
    run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
    config_dict = asdict(config)
    config_dict["algo_id"] = config.algo_id
    wandb.init(project=project_name, id=run_id, name=run_id, config=config_dict)

    for key, value in dict(wandb.config).items():
        config_dict[key] = value

    return Config.build_from_dict(config_dict)


def ensure_wandb_run_initialized(config) -> None:
    """Ensure a W&B run exists before agent construction."""
    if not getattr(config, "enable_wandb", True):
        return
    if wandb.run is not None:
        return

    project_name = config.project_id
    assert project_name, "project_id is required"
    run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb.init(project=project_name, id=run_id, name=run_id, config=asdict(config))
