#!/usr/bin/env python3
"""Multi-stage coarse-to-fine hyperparameter sweep automation.

This script implements a three-stage sweep strategy:
1. Stage 1 (30% budget): Wide grid exploration
2. Stage 2 (50% budget): Refined search around top performers
3. Stage 3 (20% budget): Fine-tuning around best config

Usage:
    # Basic usage with parameter names
    python scripts/multistage_sweep.py ALE-Pong-v5:objects_ppo \\
        --params gae_lambda,vf_coef,ent_coef \\
        --budget 500

    # Custom parameter ranges
    python scripts/multistage_sweep.py ALE-Pong-v5:objects_ppo \\
        --params "gae_lambda:0.8-0.99,vf_coef:0.1-1.0" \\
        --budget 500

    # Resume from checkpoint
    python scripts/multistage_sweep.py --resume state.json

    # Override entity/project
    python scripts/multistage_sweep.py ALE-Pong-v5:objects_ppo \\
        --params gae_lambda,vf_coef \\
        --entity myuser --project myproject \\
        --budget 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import wandb
import yaml


@dataclass
class StageResult:
    """Results from a single sweep stage."""
    stage: int
    sweep_id: str
    runs_completed: int
    best_metric: float
    best_config: Dict[str, Any]
    top_configs: List[Dict[str, Any]]


@dataclass
class MultiStageSweepState:
    """Persistent state for multi-stage sweep."""
    config_id: str
    entity: str
    project: str
    params: Dict[str, Tuple[float, float]]  # param_name -> (min, max)
    metric_name: str
    total_budget: int
    budget_used: int
    stages_completed: List[StageResult]

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'MultiStageSweepState':
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        # Convert params dict keys from strings back to tuples
        data['params'] = {k: tuple(v) for k, v in data['params'].items()}
        return cls(**data)


class MultiStageSweep:
    """Orchestrates multi-stage hyperparameter sweep."""

    def __init__(
        self,
        config_id: str,
        params: Dict[str, Tuple[float, float]],
        entity: str,
        project: str,
        total_budget: int = 500,
        metric_name: str = "train/roll/ep_rew/mean",
        state_file: Optional[Path] = None,
    ):
        self.config_id = config_id
        self.params = params
        self.entity = entity
        self.project = project
        self.total_budget = total_budget
        self.metric_name = metric_name
        self.state_file = state_file or Path(f"runs/multistage_sweep_{config_id.replace(':', '_')}/state.json")

        # Initialize state
        self.state = MultiStageSweepState(
            config_id=config_id,
            entity=entity,
            project=project,
            params=params,
            metric_name=metric_name,
            total_budget=total_budget,
            budget_used=0,
            stages_completed=[],
        )

        # Budget allocation per stage
        self.stage_budgets = {
            1: int(total_budget * 0.3),  # 30% for coarse exploration
            2: int(total_budget * 0.5),  # 50% for refinement
            3: int(total_budget * 0.2),  # 20% for fine-tuning
        }

    def create_sweep_config(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        method: str = "grid",
        grid_resolution: int = 5,
    ) -> Dict[str, Any]:
        """Create W&B sweep configuration dict.

        Args:
            param_ranges: Dict mapping parameter names to (min, max) tuples
            method: "grid" or "bayes"
            grid_resolution: Number of values per parameter for grid search

        Returns:
            Sweep config dict ready for wandb.sweep()
        """
        env_id, variant = self.config_id.split(':')

        config = {
            'name': f'{env_id}-{variant}-multistage-stage{len(self.state.stages_completed)+1}',
            'project': self.project,
            'method': method,
            'metric': {
                'name': self.metric_name,
                'goal': 'maximize',
            },
            'parameters': {},
        }

        # Add early termination for Bayesian
        if method == 'bayes':
            config['early_terminate'] = {
                'type': 'hyperband',
                'min_iter': 100,
            }

        # Build parameter specs
        for param_name, (min_val, max_val) in param_ranges.items():
            if method == 'grid':
                # Create grid of values
                import numpy as np
                values = np.linspace(min_val, max_val, grid_resolution).tolist()
                config['parameters'][param_name] = {'values': values}
            else:  # bayes
                config['parameters'][param_name] = {
                    'min': min_val,
                    'max': max_val,
                }

        # Add training command
        config['program'] = 'train.py'
        config['command'] = [
            '${env}',
            '${interpreter}',
            '${program}',
            '--config_id',
            self.config_id,
            '--wandb_sweep',
        ]

        return config

    def create_sweep(self, config: Dict[str, Any]) -> str:
        """Create W&B sweep and return sweep ID.

        Args:
            config: Sweep configuration dict

        Returns:
            Sweep ID string
        """
        print(f"\n{'='*60}")
        print(f"Creating sweep: {config['name']}")
        print(f"Method: {config['method']}")
        print(f"Parameters: {list(config['parameters'].keys())}")
        print(f"{'='*60}\n")

        sweep_id = wandb.sweep(config, entity=self.entity, project=self.project)
        print(f"✓ Sweep created: {sweep_id}")
        return sweep_id

    def launch_modal_workers(self, sweep_id: str, count: int, runs_per_worker: int = 1) -> None:
        """Launch Modal workers for sweep.

        Args:
            sweep_id: W&B sweep ID
            count: Total number of runs to execute
            runs_per_worker: Runs per worker (default: 1)
        """
        cmd = [
            'modal', 'run', 'scripts/modal_sweep_runner.py',
            '--sweep-id', sweep_id,
            '--entity', self.entity,
            '--project', self.project,
            '--count', str(count),
            '--runs-per-worker', str(runs_per_worker),
        ]

        print(f"\nLaunching {count} runs via Modal...")
        print(f"Command: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)
        print(f"✓ Modal workers launched")

    def wait_for_completion(self, sweep_id: str, expected_runs: int, poll_interval: int = 60) -> None:
        """Wait for sweep runs to complete.

        Args:
            sweep_id: W&B sweep ID
            expected_runs: Number of runs we expect to complete
            poll_interval: Seconds between status checks
        """
        api = wandb.Api()
        sweep_path = f"{self.entity}/{self.project}/{sweep_id}"

        print(f"\nWaiting for {expected_runs} runs to complete...")
        print(f"Polling every {poll_interval}s. Press Ctrl+C to abort.\n")

        last_completed = 0
        while True:
            try:
                sweep = api.sweep(sweep_path)
                runs = list(sweep.runs)

                completed = sum(1 for r in runs if r.state in ['finished', 'crashed', 'failed'])
                running = sum(1 for r in runs if r.state == 'running')

                if completed != last_completed:
                    print(f"[{time.strftime('%H:%M:%S')}] Completed: {completed}/{expected_runs}, Running: {running}")
                    last_completed = completed

                if completed >= expected_runs:
                    print(f"\n✓ All {expected_runs} runs completed!")
                    break

                time.sleep(poll_interval)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Current progress saved to state file.")
                print(f"Resume with: python {sys.argv[0]} --resume {self.state_file}")
                sys.exit(1)

    def fetch_sweep_results(self, sweep_id: str) -> List[Dict[str, Any]]:
        """Fetch all finished runs from a sweep.

        Args:
            sweep_id: W&B sweep ID

        Returns:
            List of dicts containing config and metrics for each run
        """
        api = wandb.Api()
        sweep_path = f"{self.entity}/{self.project}/{sweep_id}"
        sweep = api.sweep(sweep_path)

        results = []
        for run in sweep.runs:
            if run.state != 'finished':
                continue

            # Extract config and metric
            config = dict(run.config)
            metric_value = run.summary.get(self.metric_name)

            if metric_value is None:
                print(f"Warning: Run {run.id} missing metric {self.metric_name}, skipping")
                continue

            results.append({
                'run_id': run.id,
                'config': config,
                'metric': metric_value,
            })

        print(f"Fetched {len(results)} completed runs from sweep {sweep_id}")
        return results

    def analyze_top_configs(self, results: List[Dict[str, Any]], top_k: int = 5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Identify best config and top-K configs.

        Args:
            results: List of run results from fetch_sweep_results()
            top_k: Number of top configs to return (default: 5, minimum 10% of results)

        Returns:
            Tuple of (best_config, list_of_top_k_configs)
        """
        # Sort by metric (descending)
        sorted_results = sorted(results, key=lambda x: x['metric'], reverse=True)

        # Take at least 10% of runs or top_k, whichever is larger
        k = max(top_k, int(len(sorted_results) * 0.1))
        top_results = sorted_results[:k]

        best = top_results[0]
        print(f"\n{'='*60}")
        print(f"Top {k} configs analysis:")
        print(f"Best: {best['metric']:.3f} - {best['config']}")
        print(f"{'='*60}\n")

        return best['config'], [r['config'] for r in top_results]

    def compute_narrowed_ranges(
        self,
        top_configs: List[Dict[str, Any]],
        narrowing_factor: float = 0.5,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute narrowed parameter ranges from top configs.

        Args:
            top_configs: List of config dicts from top performers
            narrowing_factor: How much to narrow (0.5 = ±2 std, 0.25 = ±1 std)

        Returns:
            Dict mapping parameter names to new (min, max) tuples
        """
        import numpy as np

        narrowed = {}

        for param_name in self.params.keys():
            # Extract values for this param from top configs
            values = []
            for config in top_configs:
                if param_name in config:
                    values.append(config[param_name])

            if not values:
                # Param not in configs, keep original range
                narrowed[param_name] = self.params[param_name]
                continue

            # Compute statistics
            mean = np.mean(values)
            std = np.std(values)

            # Narrow range: mean ± (std / narrowing_factor)
            range_width = std / narrowing_factor
            new_min = mean - range_width
            new_max = mean + range_width

            # Clip to original bounds
            orig_min, orig_max = self.params[param_name]
            new_min = max(new_min, orig_min)
            new_max = min(new_max, orig_max)

            narrowed[param_name] = (new_min, new_max)

            print(f"{param_name}: [{orig_min:.4f}, {orig_max:.4f}] → [{new_min:.4f}, {new_max:.4f}]")

        return narrowed

    def run_stage(
        self,
        stage_num: int,
        param_ranges: Dict[str, Tuple[float, float]],
        budget: int,
        method: str = "grid",
    ) -> StageResult:
        """Run a single sweep stage.

        Args:
            stage_num: Stage number (1, 2, or 3)
            param_ranges: Parameter ranges for this stage
            budget: Number of runs for this stage
            method: "grid" or "bayes"

        Returns:
            StageResult with sweep ID and top configs
        """
        print(f"\n{'#'*60}")
        print(f"# STAGE {stage_num}: {method.upper()} SEARCH")
        print(f"# Budget: {budget} runs")
        print(f"# Parameters: {list(param_ranges.keys())}")
        print(f"{'#'*60}\n")

        # Create sweep config
        grid_resolution = 5 if stage_num == 1 else 3
        sweep_config = self.create_sweep_config(param_ranges, method, grid_resolution)

        # Create sweep in W&B
        sweep_id = self.create_sweep(sweep_config)

        # Launch Modal workers
        self.launch_modal_workers(sweep_id, budget)

        # Wait for completion
        self.wait_for_completion(sweep_id, budget)

        # Fetch results
        results = self.fetch_sweep_results(sweep_id)

        # Analyze top configs
        best_config, top_configs = self.analyze_top_configs(results)
        best_metric = max(r['metric'] for r in results)

        # Create stage result
        stage_result = StageResult(
            stage=stage_num,
            sweep_id=sweep_id,
            runs_completed=len(results),
            best_metric=best_metric,
            best_config=best_config,
            top_configs=top_configs,
        )

        # Update state
        self.state.stages_completed.append(stage_result)
        self.state.budget_used += len(results)
        self.state.save(self.state_file)

        print(f"\n✓ Stage {stage_num} complete!")
        print(f"  Best metric: {best_metric:.3f}")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Budget used: {self.state.budget_used}/{self.total_budget}\n")

        return stage_result

    def run(self) -> Dict[str, Any]:
        """Execute full multi-stage sweep.

        Returns:
            Best configuration found across all stages
        """
        print(f"\n{'='*60}")
        print(f"MULTI-STAGE SWEEP: {self.config_id}")
        print(f"Total budget: {self.total_budget} runs")
        print(f"Parameters: {list(self.params.keys())}")
        print(f"State file: {self.state_file}")
        print(f"{'='*60}\n")

        # Stage 1: Coarse exploration
        stage1 = self.run_stage(
            stage_num=1,
            param_ranges=self.params,
            budget=self.stage_budgets[1],
            method='grid',
        )

        # Stage 2: Refined search
        narrowed_ranges = self.compute_narrowed_ranges(stage1.top_configs, narrowing_factor=0.5)
        stage2 = self.run_stage(
            stage_num=2,
            param_ranges=narrowed_ranges,
            budget=self.stage_budgets[2],
            method='bayes',
        )

        # Stage 3: Fine-tuning
        very_narrow_ranges = self.compute_narrowed_ranges(stage2.top_configs, narrowing_factor=0.25)
        stage3 = self.run_stage(
            stage_num=3,
            param_ranges=very_narrow_ranges,
            budget=self.stage_budgets[3],
            method='bayes',
        )

        # Find best overall
        best_stage = max(
            [stage1, stage2, stage3],
            key=lambda s: s.best_metric,
        )

        print(f"\n{'='*60}")
        print(f"MULTI-STAGE SWEEP COMPLETE!")
        print(f"{'='*60}")
        print(f"Total runs: {self.state.budget_used}")
        print(f"Best metric: {best_stage.best_metric:.3f} (Stage {best_stage.stage})")
        print(f"Best config: {best_stage.best_config}")
        print(f"\nState saved to: {self.state_file}")
        print(f"{'='*60}\n")

        return best_stage.best_config


def parse_params_arg(params_str: str) -> Dict[str, Tuple[float, float]]:
    """Parse parameter specification string.

    Supports two formats:
    1. Simple: "param1,param2,param3" (uses default ranges)
    2. With ranges: "param1:0.8-0.99,param2:0.1-1.0"

    Args:
        params_str: Parameter specification string

    Returns:
        Dict mapping parameter names to (min, max) tuples
    """
    # Default ranges for common parameters
    default_ranges = {
        'gae_lambda': (0.8, 0.99),
        'vf_coef': (0.1, 1.0),
        'ent_coef': (0.0, 0.05),
        'policy_lr': (0.0001, 0.01),
        'clip_range': (0.1, 0.3),
        'n_steps': (64, 1024),
        'batch_size': (256, 4096),
        'gamma': (0.95, 0.999),
    }

    params = {}

    for spec in params_str.split(','):
        spec = spec.strip()

        if ':' in spec:
            # Format: param:min-max
            param_name, range_str = spec.split(':')
            min_val, max_val = map(float, range_str.split('-'))
            params[param_name] = (min_val, max_val)
        else:
            # Format: param (use default range)
            param_name = spec
            if param_name not in default_ranges:
                raise ValueError(
                    f"No default range for parameter '{param_name}'. "
                    f"Please specify as '{param_name}:min-max'"
                )
            params[param_name] = default_ranges[param_name]

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage coarse-to-fine hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'config_id',
        nargs='?',
        help='Environment:variant config (e.g., "ALE-Pong-v5:objects_ppo")',
    )
    parser.add_argument(
        '--params',
        help='Parameters to sweep (e.g., "gae_lambda,vf_coef" or "gae_lambda:0.8-0.99,vf_coef:0.1-1.0")',
    )
    parser.add_argument(
        '--budget',
        type=int,
        default=500,
        help='Total number of runs across all stages (default: 500)',
    )
    parser.add_argument(
        '--metric',
        default='train/roll/ep_rew/mean',
        help='Metric to optimize (default: train/roll/ep_rew/mean)',
    )
    parser.add_argument(
        '--entity',
        help='W&B entity (default: from WANDB_ENTITY env var)',
    )
    parser.add_argument(
        '--project',
        help='W&B project (default: from WANDB_PROJECT env var)',
    )
    parser.add_argument(
        '--resume',
        type=Path,
        help='Resume from state file',
    )

    args = parser.parse_args()

    # Handle resume
    if args.resume:
        print(f"Resuming from state file: {args.resume}")
        state = MultiStageSweepState.load(args.resume)

        sweep = MultiStageSweep(
            config_id=state.config_id,
            params=state.params,
            entity=state.entity,
            project=state.project,
            total_budget=state.total_budget,
            metric_name=state.metric_name,
            state_file=args.resume,
        )
        sweep.state = state

        # Continue from where we left off
        # TODO: Implement resume logic based on stages_completed
        print("Resume functionality not yet implemented. Please start a new sweep.")
        sys.exit(1)

    # Validate required arguments
    if not args.config_id:
        parser.error("config_id is required (unless using --resume)")
    if not args.params:
        parser.error("--params is required (unless using --resume)")

    # Parse parameters
    params = parse_params_arg(args.params)

    # Get entity/project
    entity = args.entity or os.getenv('WANDB_ENTITY')
    project = args.project or os.getenv('WANDB_PROJECT')

    if not entity:
        parser.error("--entity required or set WANDB_ENTITY environment variable")
    if not project:
        parser.error("--project required or set WANDB_PROJECT environment variable")

    # Create and run sweep
    sweep = MultiStageSweep(
        config_id=args.config_id,
        params=params,
        entity=entity,
        project=project,
        total_budget=args.budget,
        metric_name=args.metric,
    )

    best_config = sweep.run()

    # Print final summary
    print("\nBest configuration found:")
    print(json.dumps(best_config, indent=2))


if __name__ == '__main__':
    main()
