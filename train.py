import argparse

from utils.train_launcher import launch_training_from_args, list_available_environments


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument(
        "config", 
        nargs="?",
        help="Config spec '<env>:<variant>' (e.g., CartPole-v1:ppo) or environment name for fuzzy search"
    )
    parser.add_argument(
        "--config_id", 
        type=str, 
        default=None, 
        help="Config ID '<env>:<variant>' (e.g., CartPole-v1:ppo)"
    )
    parser.add_argument(
        "--wandb_sweep", 
        action="store_true", 
        default=False, 
        help="Enable W&B sweep mode: initialize wandb early and merge wandb.config into the main Config before training."
    )
    parser.add_argument(
        "--max-timesteps",
        dest="max_timesteps",
        default=None,
        help="Override config max_timesteps (total training steps). Accepts integers or scientific notation.",
    )
    parser.add_argument(
        "--list-envs",
        nargs="?",
        const="",
        metavar="SEARCH",
        help="List all available environment targets with descriptions and exit. Optionally provide a search term to filter environments (e.g., 'Pong' or 'CartPole')."
    )
    args = parser.parse_args()


    # In case list envs flag is passed, use argument as search 
    # term and show all available environments matching it
    # (exits after printing)
    config_id = args.config or args.config_id or "Bandit-v0:ppo" # TODO: move default up
    should_search = ":" not in config_id or args.list_envs is not None
    if should_search:
        search_term = args.list_envs if args.list_envs else config_id
        if not search_term: search_term = None
        list_available_environments(search_term)
        return

    # Parse args and start training
    launch_training_from_args(args)
        
if __name__ == "__main__":
    main()
