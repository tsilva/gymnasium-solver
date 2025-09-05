import os
import wandb
import argparse
from dataclasses import asdict
from utils.config import load_config, Config

def _maybe_init_wandb_sweep(config: Config):
    algo_id = config.algo_id
    config = Config.create_for_algo(algo_id, **{
        **asdict(config),
        **wandb.config,
    })
    wandb.init(config=asdict(config)) # TODO: do I need to do this here?
    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config_id", type=str, default="CartPole-v1:ppo", help="Config ID (e.g., CartPole-v1_ppo)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Run non-interactively: auto-accept prompts and defaults")
    parser.add_argument("--wandb_sweep", action="store_true", default=False, help="Enable W&B sweep mode: initialize wandb early and merge wandb.config into the main Config before training.")
    args = parser.parse_args()

    # Load configuration
    config_id = args.config_id
    config_id, variant_id = config_id.split(":")
    config = load_config(config_id, variant_id)

    # Apply args to config
    if args.quiet is True: config.quiet = True

    # If running under a W&B agent or explicitly requested, initialize wandb early
    # and merge sweep overrides into our Config before creating the agent.
    use_wandb_sweep = bool(args.wandb_sweep) or bool(os.environ.get("WANDB_SWEEP_ID") or os.environ.get("SWEEP_ID"))
    if use_wandb_sweep: config = _maybe_init_wandb_sweep(config)

    # TODO: move this out of here
    # Set global random seed
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    # Create agent and start learning
    from agents import create_agent
    agent = create_agent(config)
    agent.learn()
        
if __name__ == "__main__":
    main()
