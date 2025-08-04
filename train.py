import argparse
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1", help="Config ID (default: CartPole-v1)")
    parser.add_argument("--algo", type=str, default="ppo", help="Agent type (default: ppo)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()

    config = load_config(args.config, args.algo)
    
    # Override resume flag if provided via command line
    if args.resume:
        config.resume = True
    
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    from agents import create_agent
    agent = create_agent(config)
    print(str(agent))

    print("Starting training...")
    agent.run_training()
    print("Training completed.")
    
if __name__ == "__main__":
    main()
