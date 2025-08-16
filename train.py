import argparse

from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1_ppo", help="Config ID (default: Pong-v5_ram_ppo)")
    parser.add_argument("--algo", type=str, default=None, help="Agent type (optional, extracted from config if not provided)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    args = parser.parse_args()

    config = load_config(args.config, args.algo)
    
    # Override resume flag if provided via command line
    if args.resume:
        config.resume = True
    
    # Group session header and config/agent details neatly once
    print("=== Training Session Started ===")
    
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    from agents import create_agent
    agent = create_agent(config)

    # Print config details once here (will also be captured to logs)
    from utils.logging import log_config_details
    log_config_details(config)

    # Print model details once
    print(str(agent))
    agent.learn()
    print("Training completed.")
        
if __name__ == "__main__":
    main()
