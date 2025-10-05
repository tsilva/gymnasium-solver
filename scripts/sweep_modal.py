#!/usr/bin/env python3
"""Convenience wrapper for launching W&B sweeps on Modal AI.

This script combines sweep creation and Modal deployment into a single command.

Usage:
    # Create sweep and launch 10 workers
    python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml --count 10

    # Launch workers for existing sweep
    python scripts/sweep_modal.py --sweep-id <sweep_id> --count 20

    # Override entity/project
    python scripts/sweep_modal.py config/sweeps/cartpole_ppo_grid.yaml \\
        --entity myuser --project myproject --count 10

    # Configure worker behavior
    python scripts/sweep_modal.py --sweep-id <sweep_id> \\
        --count 50 --runs-per-worker 5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def create_sweep(sweep_config_path: str, entity: str = None, project: str = None) -> str:
    """Create W&B sweep and return sweep ID.

    Args:
        sweep_config_path: Path to sweep YAML config
        entity: W&B entity (optional, uses wandb default)
        project: W&B project (optional, uses config default)

    Returns:
        Sweep ID string
    """
    # Verify config file exists
    if not Path(sweep_config_path).exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_config_path}")

    cmd = ["wandb", "sweep"]

    if entity:
        cmd.extend(["--entity", entity])
    if project:
        cmd.extend(["--project", project])

    cmd.append(sweep_config_path)

    print(f"Creating sweep: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Print full output for debugging
    print("W&B output:")
    print(result.stdout)
    if result.stderr:
        print("W&B stderr:")
        print(result.stderr)

    # Extract sweep ID from output
    # Output format: "wandb: Created sweep with ID: abc123xyz"
    # or "wandb: View sweep at: https://wandb.ai/entity/project/sweeps/abc123xyz"
    output_text = result.stdout + "\n" + result.stderr
    for line in output_text.split("\n"):
        if "Created sweep with ID:" in line:
            sweep_id = line.split("ID:")[-1].strip()
            print(f"Extracted sweep ID: {sweep_id}")
            return sweep_id
        elif "/sweeps/" in line and "http" in line:
            # Extract from URL
            sweep_id = line.split("/sweeps/")[-1].strip()
            # Remove any trailing text after sweep ID
            sweep_id = sweep_id.split()[0].split("?")[0]
            print(f"Extracted sweep ID from URL: {sweep_id}")
            return sweep_id

    raise RuntimeError(f"Could not extract sweep ID from wandb output:\n{output_text}")


def launch_modal_workers(
    sweep_id: str,
    entity: str,
    project: str,
    count: int,
    runs_per_worker: int,
) -> None:
    """Launch Modal workers for sweep.

    Args:
        sweep_id: W&B sweep ID
        entity: W&B entity
        project: W&B project
        count: Total number of runs
        runs_per_worker: Runs per worker
    """
    cmd = [
        "modal",
        "run",
        "scripts/modal_sweep_runner.py",
        "--sweep-id",
        sweep_id,
        "--entity",
        entity,
        "--project",
        project,
        "--count",
        str(count),
        "--runs-per-worker",
        str(runs_per_worker),
    ]

    print(f"\nLaunching Modal workers: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create and launch W&B sweep on Modal AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sweep_config",
        nargs="?",
        help="Path to sweep YAML config (omit if using --sweep-id)",
    )
    parser.add_argument(
        "--sweep-id",
        help="Existing sweep ID (skips sweep creation)",
    )
    parser.add_argument(
        "--entity",
        help="W&B entity/username (default: from WANDB_ENTITY env var or wandb default)",
    )
    parser.add_argument(
        "--project",
        help="W&B project name (default: from WANDB_PROJECT env var or sweep config)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Total number of sweep runs (default: 10)",
    )
    parser.add_argument(
        "--runs-per-worker",
        type=int,
        default=1,
        help="Number of runs each Modal worker executes (default: 1)",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Only create sweep, don't launch Modal workers",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.sweep_id and not args.sweep_config:
        parser.error("Must provide either sweep_config or --sweep-id")

    # Get entity/project from environment if not provided
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    project = args.project or os.environ.get("WANDB_PROJECT")

    # Create sweep if needed
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep: {sweep_id}")
    else:
        print(f"Creating sweep from: {args.sweep_config}")
        sweep_id = create_sweep(args.sweep_config, entity, project)
        print(f"✓ Sweep created: {sweep_id}")

    # Get entity/project for Modal launch (required at this point)
    if not entity:
        raise ValueError(
            "entity must be provided via --entity flag or WANDB_ENTITY environment variable"
        )
    if not project:
        # Try to extract from sweep config if available
        if args.sweep_config:
            import yaml
            with open(args.sweep_config) as f:
                config = yaml.safe_load(f)
                project = project or config.get("project")

        if not project:
            raise ValueError(
                "project must be provided via --project flag, WANDB_PROJECT environment variable, or sweep config"
            )

    # Launch Modal workers unless --create-only
    if args.create_only:
        print(f"\nSweep created. To launch workers manually:")
        print(f"  modal run scripts/modal_sweep_runner.py \\")
        print(f"    --sweep-id {sweep_id} \\")
        print(f"    --entity {entity} \\")
        print(f"    --project {project} \\")
        print(f"    --count {args.count} \\")
        print(f"    --runs-per-worker {args.runs_per_worker}")
    else:
        launch_modal_workers(
            sweep_id=sweep_id,
            entity=entity,
            project=project,
            count=args.count,
            runs_per_worker=args.runs_per_worker,
        )
        print("\n✓ All Modal workers launched!")


if __name__ == "__main__":
    main()
