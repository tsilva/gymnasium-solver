import argparse
import os
from pathlib import Path

from utils.train_launcher import launch_training_from_args


def find_closest_match(search_term, candidates):
    """Find the closest match for a search term among candidates using fuzzy matching."""
    if not search_term:
        return None
    
    search_lower = search_term.lower()
    candidates_lower = [c.lower() for c in candidates]
    
    # Exact match first
    for i, candidate in enumerate(candidates_lower):
        if search_lower == candidate:
            return candidates[i]
    
    # Substring match
    for i, candidate in enumerate(candidates_lower):
        if search_lower in candidate or candidate in search_lower:
            return candidates[i]
    
    # Word-based matching (split on hyphens and underscores)
    search_words = set(search_lower.replace('-', ' ').replace('_', ' ').split())
    
    best_match = None
    best_score = 0
    
    for i, candidate in enumerate(candidates_lower):
        candidate_words = set(candidate.replace('-', ' ').replace('_', ' ').split())
        
        # Calculate overlap score
        overlap = len(search_words.intersection(candidate_words))
        if overlap > best_score:
            best_score = overlap
            best_match = candidates[i]
    
    return best_match if best_score > 0 else None


def list_available_environments(search_term=None, exact_match=None):
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
    
    # Get all YAML files in the config directory
    yaml_files = sorted(config_dir.glob("*.yaml"))
    env_names = [f.stem for f in yaml_files]
    
    # If exact_match provided, use it directly
    if exact_match:
        yaml_files = [f for f in yaml_files if f.stem == exact_match]
        if not yaml_files:
            print(f"Environment '{exact_match}' not found.")
            return
        print(f"{BOLD}Environment targets for '{exact_match}':{RESET}")
    # If search term provided, find closest match
    elif search_term:
        matched_env = find_closest_match(search_term, env_names)
        if not matched_env:
            print(f"No environment found matching '{search_term}'")
            print(f"Available environments: {', '.join(env_names)}")
            return
        
        # Filter to only the matched environment
        yaml_files = [f for f in yaml_files if f.stem == matched_env]
        print(f"{BOLD}Environment targets for '{matched_env}':{RESET}")
    else:
        print(f"{BOLD}Available Environment Targets:{RESET}")
    
    print()
    
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


def find_matching_environment(env_name):
    """Find the best matching environment for a given name."""
    from utils.config import Config
    from utils.io import read_yaml
    
    config_dir = Path("config/environments")
    if not config_dir.exists():
        return None
    
    # Check for exact match first
    env_file = config_dir / f"{env_name}.yaml"
    if env_file.exists():
        return env_name
    
    # Check for fuzzy match
    yaml_files = list(config_dir.glob("*.yaml"))
    env_names = [f.stem for f in yaml_files]
    matched_env = find_closest_match(env_name, env_names)
    
    return matched_env


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
        nargs="?",
        const="",
        metavar="SEARCH",
        help="List all available environment targets with descriptions and exit. Optionally provide a search term to filter environments (e.g., 'Pong' or 'CartPole')."
    )
    args = parser.parse_args()

    # Handle --list-envs flag
    if args.list_envs is not None:
        search_term = args.list_envs if args.list_envs else None
        list_available_environments(search_term)
        return

    # Handle config argument - check if it's a valid environment or needs fuzzy search
    config_spec = args.config or args.config_id or "Bandit-v0:ppo"
    
    # If it doesn't contain ':', treat it as an environment name for fuzzy search
    if ":" not in config_spec:
        # Find the best matching environment
        matched_env = find_matching_environment(config_spec)
        if not matched_env:
            print(f"No environment found matching '{config_spec}'")
            print("Available environments:")
            print()
            list_available_environments()
            return
        else:
            # Environment exists, show its targets
            print(f"Environment '{config_spec}' found. Available targets:")
            print()
            list_available_environments(exact_match=matched_env)
            return

    # Parse args and start training
    launch_training_from_args(args)
        
if __name__ == "__main__":
    main()
