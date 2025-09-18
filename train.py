import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    # Optional positional config spec to allow: `python train.py CartPole-v1:ppo`
    parser.add_argument("config", nargs="?", help="Config spec '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    # Backwards-compatible named flag
    parser.add_argument("--config_id", type=str, default=None, help="Config ID '<env>:<variant>' (e.g., CartPole-v1:ppo)")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Run non-interactively: auto-accept prompts and defaults")
    parser.add_argument("--wandb_sweep", action="store_true", default=False, help="Enable W&B sweep mode: initialize wandb early and merge wandb.config into the main Config before training.")
    parser.add_argument(
        "--max-steps",
        "--max-timesteps",
        "--max_timesteps",
        dest="max_timesteps",
        default=None,
        help="Override config max_timesteps (total training steps). Accepts integers or scientific notation.",
    )
    args = parser.parse_args()

    # Delegate end-to-end training to the launcher
    from utils.train_launcher import launch_training_from_args

    launch_training_from_args(args)
        
if __name__ == "__main__":
    main()
