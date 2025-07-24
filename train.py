import argparse

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--agent", type=str, default="ppo", help="Agent type (default: ppo)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium environment (default: CartPole-v1)")
    # TODO: set to 1 if in debug mode
    parser.add_argument("--n_envs", type=str, default=1, help="Number of environments (default: auto)")
    args = parser.parse_args()

    from agents import create_agent
    agent = create_agent(args.agent, args.env, n_envs=args.n_envs)
    agent.train()
    
if __name__ == "__main__":
    main()
