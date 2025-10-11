#!/usr/bin/env python3
"""Convenience wrapper for launching simple grid sweeps on Modal AI.

This script creates a grid sweep from command-line parameter specs and launches
Modal workers. No YAML config files needed.

Usage:
    # Sweep single parameter with discrete values
    python scripts/sweep_grid.py CartPole-v1:ppo \\
        --params "policy_lr:0.001,0.003,0.01"

    # Sweep multiple parameters (cartesian product)
    python scripts/sweep_grid.py ALE-Pong-v5:ppo \\
        --params "gae_lambda:0.90,0.95,0.99" "vf_coef:0.25,0.5,1.0"

    # Override entity/project and worker config
    python scripts/sweep_grid.py ALE-Pong-v5:ppo \\
        --params "gae_lambda:0.90,0.95,0.99" \\
        --entity myuser --project myproject \\
        --runs-per-worker 3

    # Create sweep but don't launch workers
    python scripts/sweep_grid.py CartPole-v1:ppo \\
        --params "policy_lr:0.001,0.003,0.01" \\
        --create-only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import wandb
from utils.config import load_config
from utils.formatting import sanitize_name


def parse_param_spec(spec: str) -> tuple[str, List[float]]:
    """Parse parameter spec like 'gae_lambda:0.85,0.90,0.95'.

    Args:
        spec: Parameter spec string

    Returns:
        Tuple of (param_name, list_of_values)
    """
    if ':' not in spec:
        raise ValueError(f"Parameter spec must be 'param_name:val1,val2,...' but got: {spec}")

    param_name, values_str = spec.split(':', 1)
    param_name = param_name.strip()

    # Parse comma-separated values
    values = []
    for v in values_str.split(','):
        v = v.strip()
        try:
            values.append(float(v))
        except ValueError:
            raise ValueError(f"Invalid value '{v}' in parameter spec: {spec}")

    if not values:
        raise ValueError(f"No values provided in parameter spec: {spec}")

    return param_name, values


def create_grid_sweep_config(
    config_id: str,
    params: Dict[str, List[float]],
    project: str,
    metric_name: str = "train/roll/ep_rew/mean",
) -> Dict[str, Any]:
    """Create W&B sweep configuration for grid search.

    Args:
        config_id: Environment:variant config (e.g., "CartPole-v1:ppo")
        params: Dict mapping parameter names to lists of discrete values
        project: W&B project name
        metric_name: Metric to optimize

    Returns:
        Sweep config dict ready for wandb.sweep()
    """
    env_id, variant = config_id.split(':')

    config = {
        'name': f'{env_id}-{variant}-grid',
        'project': project,
        'method': 'grid',
        'metric': {
            'name': metric_name,
            'goal': 'maximize',
        },
        'parameters': {},
    }

    # Build parameter specs
    for param_name, values in params.items():
        config['parameters'][param_name] = {'values': values}

    # Add training command
    config['program'] = 'train.py'
    config['command'] = [
        '${env}',
        '${interpreter}',
        '${program}',
        '--config_id',
        config_id,
        '--wandb_sweep',
    ]

    return config


def create_sweep(config: Dict[str, Any], entity: str, project: str) -> str:
    """Create W&B sweep and return sweep ID.

    Args:
        config: Sweep configuration dict
        entity: W&B entity
        project: W&B project

    Returns:
        Sweep ID string
    """
    print(f"\n{'='*60}")
    print(f"Creating grid sweep: {config['name']}")
    print(f"Parameters:")
    for param_name, spec in config['parameters'].items():
        values = spec['values']
        print(f"  {param_name}: {values} ({len(values)} values)")

    # Calculate total runs
    total_runs = 1
    for spec in config['parameters'].values():
        total_runs *= len(spec['values'])
    print(f"\nTotal runs: {total_runs}")
    print(f"{'='*60}\n")

    sweep_id = wandb.sweep(config, entity=entity, project=project)
    print(f"✓ Sweep created: {sweep_id}")
    print(f"  View at: https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")

    return sweep_id


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
    print("\n✓ All Modal workers launched!")


def main():
    parser = argparse.ArgumentParser(
        description="Create and launch simple grid sweep on Modal AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "config_id",
        help='Environment:variant config (e.g., "CartPole-v1:ppo")',
    )
    parser.add_argument(
        "--params",
        nargs="+",
        required=True,
        help='Parameter specs like "param:val1,val2,..." (can specify multiple)',
    )
    parser.add_argument(
        "--metric",
        default="train/roll/ep_rew/mean",
        help="Metric to optimize (default: train/roll/ep_rew/mean)",
    )
    parser.add_argument(
        "--entity",
        help="W&B entity (default: from WANDB_ENTITY env var or .env file)",
    )
    parser.add_argument(
        "--project",
        help="W&B project (default: inferred from config's project_id, or WANDB_PROJECT env var)",
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

    # Parse parameter specs
    params = {}
    for spec in args.params:
        param_name, values = parse_param_spec(spec)
        params[param_name] = values

    # Get entity from env (required)
    entity = args.entity or os.environ.get("WANDB_ENTITY")
    if not entity:
        parser.error("--entity required or set WANDB_ENTITY environment variable")

    # Infer project from config if not explicitly provided
    project = args.project or os.environ.get("WANDB_PROJECT")
    if not project:
        # Load config to get project_id
        env_id, variant_id = args.config_id.split(':')
        config = load_config(env_id, variant_id)
        project = sanitize_name(config.project_id)
        print(f"Inferred project from config: {project}")

    # Create sweep config
    sweep_config = create_grid_sweep_config(
        config_id=args.config_id,
        params=params,
        project=project,
        metric_name=args.metric,
    )

    # Create sweep in W&B
    sweep_id = create_sweep(sweep_config, entity, project)

    # Calculate total runs (cartesian product)
    total_runs = 1
    for values in params.values():
        total_runs *= len(values)

    # Launch Modal workers unless --create-only
    if args.create_only:
        print(f"\nSweep created. To launch workers manually:")
        print(f"  modal run scripts/modal_sweep_runner.py \\")
        print(f"    --sweep-id {sweep_id} \\")
        print(f"    --entity {entity} \\")
        print(f"    --project {project} \\")
        print(f"    --count {total_runs} \\")
        print(f"    --runs-per-worker {args.runs_per_worker}")
    else:
        launch_modal_workers(
            sweep_id=sweep_id,
            entity=entity,
            project=project,
            count=total_runs,
            runs_per_worker=args.runs_per_worker,
        )


if __name__ == "__main__":
    main()
