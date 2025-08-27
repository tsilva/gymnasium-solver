import argparse

from utils.config import load_config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    # New: support positional usage: `python train.py <env_or_config> [variant]`
    parser.add_argument("env_or_config", nargs="?", default=None, help="Environment or config ID (e.g., LunarLander-v3 or LunarLander-v3_ppo)")
    parser.add_argument("variant", nargs="?", default=None, help="Optional variant/algo from the YAML file (e.g., ppo)")
    # Backward-compatible flags
    parser.add_argument("--config", type=str, default=None, help="Config ID (e.g., CartPole-v1_ppo)")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm/variant (optional; used with env-only configs)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    parser.add_argument("--quiet", "-q", action="store_true", default=True, help="Run non-interactively: auto-accept prompts and defaults")
    args = parser.parse_args()

    # Resolve config selection
    config_id = None
    algo_id = None
    if args.env_or_config is not None:
        # Positional mode
        config_id = args.env_or_config
        algo_id = args.variant or args.algo
    elif args.config is not None:
        # Flag mode (legacy)
        config_id = args.config
        algo_id = args.algo
    else:
        # Fallback to legacy default
        config_id = "CartPole-v1_reinforce"
        algo_id = None

    # Load configuration (supports env-only + optional variant, or full config id)
    config = load_config(config_id, algo_id)
    
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
