import os
import wandb
import argparse
from dataclasses import asdict
from utils.config import load_config, Config

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
    args = parser.parse_args()

    # Resolve configuration spec from positional, then flag, then default
    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    if ":" not in config_spec:
        raise SystemExit("Config spec must be '<env>:<variant>' (e.g., CartPole-v1:ppo)")

    # Load configuration
    env_id, variant_id = config_spec.split(":", 1)
    config = load_config(env_id, variant_id)

    # Apply args to config
    if args.quiet is True: config.quiet = True

    # If running under a W&B agent or explicitly requested, initialize wandb early
    # and merge sweep overrides into our Config before creating the agent.
    use_wandb_sweep = bool(args.wandb_sweep) or bool(os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID"))
    if use_wandb_sweep: config = _init_wandb_sweep(config)

    # TODO: move this out of here
    # Set global random seed
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    # Create agent and start learning
    from agents import build_agent
    agent = build_agent(config)
    agent.learn()

    # Post-training: ensure a W&B workspace exists for this project and print its URL
    try:
        from utils.wandb_workspace import create_or_update_workspace_for_current_run

        url = create_or_update_workspace_for_current_run(overwrite=True, select_current_run_only=True)
        if url:
            print(f"W&B Workspace: {url}")
    except ImportError:
        # Soft dependency missing; skip silently
        pass
    except Exception as e:
        # Non-fatal: print a brief note and continue
        print(f"Warning: could not create/update W&B workspace ({e})")
        
if __name__ == "__main__":
    main()
