import argparse

from utils.config import load_config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config_id", type=str, default="ALE-Pong-v5_rgb:reinforce", help="Config ID (e.g., CartPole-v1_ppo)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Run non-interactively: auto-accept prompts and defaults")
    args = parser.parse_args()

    # Load configuration
    config_id = args.config_id
    config_id, variant_id = config_id.split(":")
    config = load_config(config_id, variant_id)
    
    # Apply args to config
    if args.quiet is True: config.quiet = True

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
