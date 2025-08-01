import argparse
import debugpy
from utils.config import load_config

def is_debugger_attached():
    return debugpy.is_client_connected()

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1", help="Config ID (default: CartPole-v1)")
    parser.add_argument("--algo", type=str, default="ppo", help="Agent type (default: ppo)")
    args = parser.parse_args()

    config = load_config(args.config, args.algo)
    
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
