"""Environment registry and listing utilities.

Provides fuzzy search and enumeration of available environment
configurations from YAML files.
"""

from __future__ import annotations

from pathlib import Path


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

    # Assert that config/environments directory exists
    config_dir = Path("config/environments")
    if not config_dir.exists():
        raise FileNotFoundError("config/environments directory not found")

    # List all env names
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
