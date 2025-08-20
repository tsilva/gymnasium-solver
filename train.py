import argparse

from utils.config import load_config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1_ppo", help="Config ID (default: CartPole-v1_ppo)")
    parser.add_argument("--algo", type=str, default=None, help="Agent type (optional, extracted from config if not provided)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run non-interactively: auto-accept prompts and defaults")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.algo)
    
    # Apply args to config
    if args.quiet is True: config.quiet = True
    if args.resume is True: config.resume = True

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
