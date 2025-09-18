"""Training launcher utilities.

Encapsulates train.py logic so the entrypoint stays minimal.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional


def _parse_positive_int(value: str, flag: str) -> int:
    """Parse a positive integer from CLI strings (supports 1e5 and 1_000 forms).

    Raises SystemExit with a friendly message on invalid input to match
    typical CLI behavior in train.py.
    """
    sanitized = str(value).replace("_", "")
    try:
        numeric = float(sanitized)
    except ValueError as exc:
        raise SystemExit(f"{flag} must be a positive integer (got '{value}').") from exc
    if numeric <= 0:
        raise SystemExit(f"{flag} must be greater than zero (got {value}).")
    if not numeric.is_integer():
        raise SystemExit(f"{flag} must be a whole number (got {value}).")
    return int(numeric)


def _init_wandb_sweep(config):
    """Initialize W&B early for sweeps and merge wandb.config into config.

    Returns a Config instance with overrides applied.
    """
    # Import locally to keep module import light for tests and non-W&B flows
    import wandb
    from utils.config import Config

    base = asdict(config)
    wandb.init(config=base)
    merged = dict(wandb.config)
    return Config.build_from_dict(merged)


def _maybe_merge_wandb_config(config, *, wandb_sweep_flag: bool, cli_max_timesteps: Optional[int]):
    """Optionally merge W&B sweep overrides into Config.

    Honors explicit --wandb_sweep or auto-detects via WANDB_SWEEP_ID/SWEEP_ID.
    Re-applies CLI --max-steps after merge to ensure CLI takes precedence.
    """
    use_wandb_sweep = bool(wandb_sweep_flag) or bool(
        os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID")
    )
    if not use_wandb_sweep:
        return config

    config = _init_wandb_sweep(config)
    if cli_max_timesteps is not None:
        config.max_timesteps = cli_max_timesteps
    return config


def launch_training_from_args(args) -> None:
    """End-to-end training launcher extracted from train.py.

    Keeps the entrypoint focused on argument parsing while this
    function handles config resolution, seeding, agent setup, and
    post-training reporting.
    """
    from stable_baselines3.common.utils import set_random_seed
    from utils.config import load_config
    from utils.formatting import format_duration
    from utils.wandb_workspace import create_or_update_workspace_for_current_run

    # Resolve configuration spec from positional, then flag, then default
    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    if ":" not in config_spec:
        raise SystemExit("Config spec must be '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    env_id, variant_id = config_spec.split(":", 1)

    # Parse CLI max-steps override early (after arg parsing) so we can apply it
    cli_max_timesteps = None
    if getattr(args, "max_timesteps", None) is not None:
        cli_max_timesteps = _parse_positive_int(str(args.max_timesteps), "--max-steps")

    # Load configuration and apply simple CLI overrides
    config = load_config(env_id, variant_id)
    if getattr(args, "quiet", False) is True:
        config.quiet = True
    if cli_max_timesteps is not None:
        config.max_timesteps = cli_max_timesteps

    # Merge W&B sweep overrides when requested/auto-detected
    config = _maybe_merge_wandb_config(
        config, wandb_sweep_flag=bool(getattr(args, "wandb_sweep", False)), cli_max_timesteps=cli_max_timesteps
    )

    # Set global RNG seed for reproducibility
    set_random_seed(config.seed)

    # Create the agent and kick off learning
    from agents import build_agent

    agent = build_agent(config)
    agent.learn()

    # Ensure a W&B workspace exists for this project and print its URL
    url = create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True)
    if url:
        print(f"W&B Workspace: {url}")

    # Print final training completion message (duration and normalized reason)
    try:
        elapsed = getattr(agent, "_fit_elapsed_seconds", None)
        if elapsed is None:
            elapsed = agent.timings.seconds_since("on_fit_start")
        human = format_duration(float(elapsed))
    except Exception:
        human = "unknown"
    reason = (
        getattr(agent, "_final_stop_reason", None)
        or getattr(agent, "_early_stop_reason", None)
        or "completed."
    )
    if isinstance(reason, str) and reason and not str(reason).endswith(".") and reason != "completed.":
        reason = f"{reason}."
    print(f"Training completed in {human}. Reason: {reason}")

