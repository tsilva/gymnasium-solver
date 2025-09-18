import argparse
import os
from dataclasses import asdict

from stable_baselines3.common.utils import set_random_seed

import wandb
from utils.config import Config, load_config
from utils.formatting import format_duration


def _parse_positive_int(value: str, flag: str) -> int:
    """Parse CLI integers while supporting scientific notation and underscores."""
    sanitized = value.replace("_", "")
    try:
        numeric = float(sanitized)
    except ValueError as exc:
        raise SystemExit(f"{flag} must be a positive integer (got '{value}').") from exc
    if numeric <= 0:
        raise SystemExit(f"{flag} must be greater than zero (got {value}).")
    if not numeric.is_integer():
        raise SystemExit(f"{flag} must be a whole number (got {value}).")
    return int(numeric)


def _init_wandb_sweep(config: Config):
    base = asdict(config)
    wandb.init(config=base)
    merged = dict(wandb.config)
    return Config.build_from_dict(merged)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    # Optional positional config spec to allow: `python train.py CartPole-v1:ppo`
    parser.add_argument("config", nargs="?", help="Config spec '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    # Backwards-compatible named flag
    parser.add_argument("--config_id", type=str, default=None, help="Config ID '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Run non-interactively: auto-accept prompts and defaults")
    parser.add_argument("--wandb_sweep", action="store_true", default=False, help="Enable W&B sweep mode: initialize wandb early and merge wandb.config into the main Config before training.")
    parser.add_argument(
        "--max-steps",
        "--max-timesteps",
        "--max_timesteps",
        dest="max_timesteps",
        default=None,
        help="Override config max_timesteps (total training steps). Accepts integers or scientific notation.",
    )
    args = parser.parse_args()

    # Resolve configuration spec from positional, then flag, then default
    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    if ":" not in config_spec:
        raise SystemExit("Config spec must be '<env>:<variant>' (e.g., CartPole-v1:ppo)")

    cli_max_timesteps = None
    if args.max_timesteps is not None:
        cli_max_timesteps = _parse_positive_int(str(args.max_timesteps), "--max-steps")

    # Load configuration
    env_id, variant_id = config_spec.split(":", 1)
    config = load_config(env_id, variant_id)

    # Apply args to config
    if args.quiet is True: config.quiet = True
    if cli_max_timesteps is not None: config.max_timesteps = cli_max_timesteps

    # If running under a W&B agent or explicitly requested, initialize wandb early
    # and merge sweep overrides into our Config before creating the agent.
    use_wandb_sweep = bool(args.wandb_sweep) or bool(os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID"))
    if use_wandb_sweep:
        config = _init_wandb_sweep(config)
        if cli_max_timesteps is not None:
            config.max_timesteps = cli_max_timesteps

    # Set global random seed
    set_random_seed(config.seed)

    # Create agent and start learning
    from agents import build_agent
    agent = build_agent(config)
    agent.learn()

    # TODO: move to callback
    # Post-training: ensure a W&B workspace exists for this project and print its URL
    from utils.wandb_workspace import create_or_update_workspace_for_current_run
    url = create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True)
    if url: print(f"W&B Workspace: {url}")
    
    # TODO: remove finish and check if it still works
    # Intentionally avoid calling wandb.finish() to suppress W&B's
    # end-of-run console summary ("Run history", etc.). The run
    # will be finalized implicitly on process exit.
    #wandb.finish()

    # TODO: move to callback
    # Print the training completion message last, after any W&B output
    try:
        # Prefer the value captured by the agent at fit end
        elapsed = getattr(agent, "_fit_elapsed_seconds", None)
        if elapsed is None:
            elapsed = agent.timings.seconds_since("on_fit_start")
        human = format_duration(float(elapsed))
    except Exception:
        human = "unknown"
    # Prefer the normalized final stop reason determined at fit end
    reason = getattr(agent, "_final_stop_reason", None) or getattr(agent, "_early_stop_reason", None) or "completed."
    # Ensure a trailing period for fallback reason
    if isinstance(reason, str) and reason and not str(reason).endswith(".") and reason != "completed.":
        reason = f"{reason}."
    print(f"Training completed in {human}. Reason: {reason}")
        
if __name__ == "__main__":
    main()
