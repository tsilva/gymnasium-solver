import argparse
import debugpy

def is_debugger_attached():
    return debugpy.is_client_connected()

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--agent", type=str, default="ppo", help="Agent type (default: ppo)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment (default: CartPole-v1)")
    args = parser.parse_args()

    from agents import create_agent
    agent = create_agent(args.agent, args.env)
    print(str(agent))
    agent.train()
    
if __name__ == "__main__":
    main()
