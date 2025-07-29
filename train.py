import argparse
import debugpy
from utils.config import load_config

def is_debugger_attached():
    return debugpy.is_client_connected()

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium environment (default: CartPole-v1)")
    parser.add_argument("--algo_id", type=str, default="ppo", help="Agent type (default: ppo)")
    args = parser.parse_args()

    config = load_config(args.env_id, args.algo_id)
    
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(config.seed)

    from agents import create_agent
    agent = create_agent(config)
    print(str(agent))

    print("Starting training...")
    agent.train()
    print("Training completed.")

    print("Starting evaluation...")
    agent.eval()
    print("Evaluation completed.")
    
if __name__ == "__main__":
    main()
