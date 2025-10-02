"""Smoke test all environment configurations by training each for N epochs.

Usage:
    python scripts/smoke_all_configs.py                    # default 2 epochs per config
    python scripts/smoke_all_configs.py --epochs 5         # 5 epochs per config
    python scripts/smoke_all_configs.py --filter CartPole  # only configs matching 'CartPole'
    python scripts/smoke_all_configs.py --limit 3          # stop after 3 configs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import build_agent
from utils.config import Config, load_config
from utils.io import read_yaml
from utils.random import set_random_seed


def discover_all_configs() -> list[tuple[str, str]]:
    """Discover all (env_id, variant_id) pairs from config/environments/*.yaml.

    Returns:
        List of (env_id, variant_id) tuples sorted by env_id then variant_id.
    """
    config_dir = Path("config/environments")
    assert config_dir.exists(), "config/environments directory not found"

    yaml_files = sorted(config_dir.glob("*.yaml"))
    config_field_names = set(Config.__dataclass_fields__.keys())

    all_configs = []
    for yaml_file in yaml_files:
        doc = read_yaml(yaml_file) or {}
        env_name = yaml_file.stem

        # Find all public targets (non-underscore keys that are dictionaries)
        for key, value in doc.items():
            # Skip base config fields and non-dict fields
            if key in config_field_names or not isinstance(value, dict):
                continue
            # Skip meta/utility sections prefixed with underscore
            if isinstance(key, str) and key.startswith("_"):
                continue
            # This is a public target
            all_configs.append((env_name, key))

    return sorted(all_configs)


def smoke_test_config(env_id: str, variant_id: str, n_epochs: int) -> tuple[bool, Optional[str]]:
    """Train a single config for N epochs.

    Returns:
        (success, error_message) tuple. success=True if training completes without exception.
    """
    try:
        # Load config
        config = load_config(env_id, variant_id)

        # Disable W&B for smoke tests (avoid polluting workspace)
        config.enable_wandb = False
        config.quiet = True

        # Force n_envs=2 and sync vectorization for faster smoke tests
        config.n_envs = 2
        config.vectorization_mode = 'sync'

        # Use fractional batch size to ensure it divides the new rollout size
        config.batch_size = 0.5  # 50% of rollout size
        config._resolve_batch_size()  # Re-resolve batch_size after changing n_envs

        # Disable evaluation for smoke tests (faster and avoids eval-related issues)
        config.eval_freq_epochs = None

        # Override to run for only N epochs
        # We'll use max_env_steps to control duration indirectly via early stopping
        # But simpler: just set very low n_epochs or max_eval_episodes
        # Actually, let's just override max_env_steps to be small
        config.max_env_steps = config.n_envs * config.n_steps * n_epochs

        # Set global seed
        set_random_seed(config.seed)

        # Build agent and train
        agent = build_agent(config)
        agent.learn()

        return True, None

    except Exception as e:
        import traceback
        return False, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test all environment configurations by training each for N epochs."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs to train each config (default: 2)"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter configs by substring match in env_id (e.g., 'CartPole', 'Pong')"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after testing N configs (useful for quick checks)"
    )

    args = parser.parse_args()

    # Discover all configs
    all_configs = discover_all_configs()

    # Apply filter if provided
    if args.filter:
        filter_lower = args.filter.lower()
        all_configs = [
            (env_id, variant_id)
            for env_id, variant_id in all_configs
            if filter_lower in env_id.lower()
        ]

    # Apply limit if provided
    if args.limit and args.limit > 0:
        all_configs = all_configs[:args.limit]

    if not all_configs:
        print("No configs found matching criteria.")
        return

    print(f"Found {len(all_configs)} config(s) to smoke test (running {args.epochs} epochs each):\n")

    # Track results
    results = []

    for i, (env_id, variant_id) in enumerate(all_configs, 1):
        config_spec = f"{env_id}:{variant_id}"
        print(f"[{i}/{len(all_configs)}] Testing {config_spec}...", end=" ", flush=True)

        success, error = smoke_test_config(env_id, variant_id, args.epochs)

        if success:
            print("✓")
            results.append((config_spec, True, None))
        else:
            print(f"✗ {error}")
            results.append((config_spec, False, error))

    # Summary
    print("\n" + "=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}\n")

    if failed > 0:
        print("Failed configs:")
        for config_spec, success, error in results:
            if not success:
                print(f"  ✗ {config_spec}")
                print(f"    Error: {error}")
        sys.exit(1)
    else:
        print("All configs passed! ✓")


if __name__ == "__main__":
    main()
