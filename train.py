import argparse
import os
from pathlib import Path

from utils.train_launcher import launch_training_from_args


def list_available_environments():
    """List all available environment targets with their descriptions."""
    from utils.config import Config
    from utils.io import read_yaml
    
    # ANSI escape codes for styling
    BOLD = '\033[1m'
    RESET = '\033[0m'
    BULLET = '•'
    
    config_dir = Path("config/environments")
    if not config_dir.exists():
        print("No environment configurations found.")
        return
    
    print(f"{BOLD}Available Environment Targets:{RESET}")
    print()
    
    # Get all YAML files in the config directory
    yaml_files = sorted(config_dir.glob("*.yaml"))
    
    for yaml_file in yaml_files:
        # Load the YAML file
        doc = read_yaml(yaml_file) or {}
        
        # Get the environment name from the filename
        env_name = yaml_file.stem
        
        # Find all public targets (non-underscore keys that are dictionaries)
        config_field_names = set(Config.__dataclass_fields__.keys())
        public_targets = []
        
        for key, value in doc.items():
            # Skip base config fields and non-dict fields
            if key in config_field_names or not isinstance(value, dict):
                continue
                
            # Skip meta/utility sections (e.g., anchors) prefixed with underscore
            if isinstance(key, str) and key.startswith("_"):
                continue
                
            # This is a public target
            description = value.get("description", "No description available")
            public_targets.append((key, description))
        
        if public_targets:
            # Use bold formatting for environment name
            print(f"{BOLD}{env_name}:{RESET}")
            for target, description in sorted(public_targets):
                print(f"  {BULLET} {env_name}:{target} - {description}")
            print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument(
        "config", 
        nargs="?",
        help="Config spec '<env>:<variant>' (e.g., CartPole-v1:ppo)"
    )
    parser.add_argument(
        "--config_id", 
        type=str, 
        default=None, 
        help="Config ID '<env>:<variant>' (e.g., CartPole-v1:ppo)"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        default=False, 
        help="Run non-interactively: auto-accept prompts and defaults"
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
        action="store_true",
        help="List all available environment targets with descriptions and exit"
    )
    args = parser.parse_args()

    # Handle --list-envs flag
    if args.list_envs:
        list_available_environments()
        return

    # Parse args and start training
    launch_training_from_args(args)
        
if __name__ == "__main__":
    main()
