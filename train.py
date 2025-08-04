import argparse
from utils.config import load_config
from utils.logging import capture_all_output, log_config_details

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1", help="Config ID (default: CartPole-v1)")
    parser.add_argument("--algo", type=str, default="ppo", help="Agent type (default: ppo)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    args = parser.parse_args()

    config = load_config(args.config, args.algo)
    
    # Override resume flag if provided via command line
    if args.resume:
        config.resume = True
    
    # Set up comprehensive logging - all output will go to both console and log file
    with capture_all_output(config=config, log_dir=args.log_dir):
        print(f"=== Training Session Started ===")
        print(f"Command: {' '.join(['python'] + [arg for arg in [args.config, '--algo', args.algo] + (['--resume'] if args.resume else [])])}")
        
        # Log configuration details
        log_config_details(config)
        
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(config.seed)

        from agents import create_agent
        agent = create_agent(config)
        print(str(agent))

        print("Starting training...")
        agent._run_training()
        print("Training completed.")
    
if __name__ == "__main__":
    main()
