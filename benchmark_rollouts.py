#!/usr/bin/env python3
"""
Benchmark script for the collect_rollouts function.

This script measures performance of the rollout collection process
across different environments, model sizes, and configurations.

Usage:
    # Run a single benchmark
    python benchmark_rollouts.py --env CartPole-v1 --n-envs 4 --n-steps 1024

    # Run the full benchmark suite
    python benchmark_rollouts.py --suite

    # Custom configuration
    python benchmark_rollouts.py --env CartPole-v1 --n-envs 8 --n-steps 2048 \
                                  --hidden-dims 128 128 --runs 5

Available options:
    --env: Environment ID (default: CartPole-v1)
    --n-envs: Number of parallel environments (default: 4)
    --n-steps: Number of steps per rollout (default: 1024)
    --hidden-dims: Hidden layer dimensions (default: [64, 64])
    --runs: Number of benchmark runs (default: 3)
    --deterministic: Use deterministic policy
    --suite: Run full benchmark suite
    --collect-frames: Collect environment frames (adds overhead)

The script outputs:
    - Execution time statistics (mean, std, min, max)
    - Throughput metrics (steps/sec, episodes/sec)
    - Environment and model information
    - Trajectory validation
"""

import time
import torch
import numpy as np
import argparse
import sys
from typing import Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.rollouts import collect_rollouts
from utils.models import PolicyNet, ValueNet
from utils.environment import _build_env


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    env_id: str
    n_envs: int
    n_steps: int
    hidden_dims: List[int]
    deterministic: bool = False
    gamma: float = 0.99
    lam: float = 0.95
    normalize_advantage: bool = True
    collect_frames: bool = False


@contextmanager
def timer():
    """Context manager for timing code execution."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")


def create_models(obs_dim: int, action_dim: int, hidden_dims: List[int], device: torch.device):
    """Create policy and value models for benchmarking."""
    policy_model = PolicyNet(obs_dim, action_dim, hidden_dims).to(device)
    value_model = ValueNet(obs_dim, hidden_dims).to(device)
    
    # Set models to eval mode for rollouts
    policy_model.eval()
    value_model.eval()
    
    return policy_model, value_model


def benchmark_collect_rollouts(config: BenchmarkConfig, device: torch.device, num_runs: int = 3) -> Dict[str, Any]:
    """Benchmark the collect_rollouts function with given configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.env_id}")
    print(f"Environment: {config.n_envs} envs, {config.n_steps} steps")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Create environment
    env = _build_env(config.env_id, n_envs=config.n_envs, seed=42)
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Observation space: {obs_dim}")
    print(f"Action space: {action_dim}")
    
    # Create models
    policy_model, value_model = create_models(obs_dim, action_dim, config.hidden_dims, device)
    
    # Warmup run
    print("\nWarmup run...")
    try:
        rollout_gen = collect_rollouts(
            env=env,
            policy_model=policy_model,
            value_model=value_model,
            n_steps=min(config.n_steps, 256),  # Small warmup
            deterministic=config.deterministic,
            gamma=config.gamma,
            lam=config.lam,
            normalize_advantage=config.normalize_advantage,
            collect_frames=config.collect_frames
        )
        next(rollout_gen)
        print("Warmup completed successfully")
    except Exception as e:
        print(f"Warmup failed: {e}")
        env.close()
        return {"error": str(e)}
    
    # Benchmark runs
    times = []
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}:")
        
        # Reset environment for consistent starting conditions
        env.reset()
        
        start_time = time.perf_counter()
        
        try:
            rollout_gen = collect_rollouts(
                env=env,
                policy_model=policy_model,
                value_model=value_model,
                n_steps=config.n_steps,
                deterministic=config.deterministic,
                gamma=config.gamma,
                lam=config.lam,
                normalize_advantage=config.normalize_advantage,
                collect_frames=config.collect_frames
            )
            
            trajectories, stats = next(rollout_gen)
            
            end_time = time.perf_counter()
            run_time = end_time - start_time
            times.append(run_time)
            results.append(stats)
            
            print(f"  Time: {run_time:.4f}s")
            print(f"  Episodes: {stats['n_episodes']}")
            print(f"  Steps: {stats['n_steps']}")
            print(f"  Mean reward: {stats['mean_ep_reward']:.3f}")
            print(f"  Mean length: {stats['mean_ep_length']:.1f}")
            
            # Validate trajectory shapes
            states, actions, rewards, dones, logps, values, advantages, returns, frames = trajectories
            expected_size = config.n_envs * (config.n_steps // config.n_envs)
            
            print(f"  Trajectory size: {len(states)} (expected: {expected_size})")
            
        except Exception as e:
            print(f"  Error in run {run + 1}: {e}")
            env.close()
            return {"error": str(e)}
    
    env.close()
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate throughput metrics
    steps_per_second = config.n_steps / mean_time
    episodes_per_second = np.mean([r['n_episodes'] for r in results]) / mean_time
    
    benchmark_results = {
        "config": config,
        "device": str(device),
        "num_runs": num_runs,
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "steps_per_second": steps_per_second,
        "episodes_per_second": episodes_per_second,
        "stats": results
    }
    
    print(f"\nResults Summary:")
    print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f}s")
    print(f"  Min time: {min_time:.4f}s")
    print(f"  Max time: {max_time:.4f}s")
    print(f"  Steps/sec: {steps_per_second:.1f}")
    print(f"  Episodes/sec: {episodes_per_second:.1f}")
    
    return benchmark_results


def run_benchmark_suite():
    """Run a comprehensive benchmark suite."""
    # Detect available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Define benchmark configurations
    configs = [
        # Small-scale benchmarks
        BenchmarkConfig(
            env_id="CartPole-v1",
            n_envs=1,
            n_steps=512,
            hidden_dims=[32]
        ),
        BenchmarkConfig(
            env_id="CartPole-v1",
            n_envs=4,
            n_steps=1024,
            hidden_dims=[32]
        ),
        BenchmarkConfig(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=2048,
            hidden_dims=[64, 64]
        ),
        
        # Medium-scale benchmarks with different environments
        BenchmarkConfig(
            env_id="MountainCar-v0",
            n_envs=4,
            n_steps=1024,
            hidden_dims=[64, 64]
        ),
        BenchmarkConfig(
            env_id="Acrobot-v1",
            n_envs=8,
            n_steps=2048,
            hidden_dims=[128, 128]
        ),
        
        # Larger model benchmark
        BenchmarkConfig(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=2048,
            hidden_dims=[256, 256, 256]
        ),
    ]
    
    all_results = []
    
    print("Starting Rollout Collection Benchmark Suite")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    for i, config in enumerate(configs):
        print(f"\n\nBenchmark {i + 1}/{len(configs)}")
        
        try:
            results = benchmark_collect_rollouts(config, device, num_runs=3)
            all_results.append(results)
        except Exception as e:
            print(f"Benchmark failed: {e}")
            continue
    
    # Print summary
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUITE SUMMARY")
    print(f"{'='*80}")
    
    for i, results in enumerate(all_results):
        if "error" in results:
            print(f"Benchmark {i+1}: FAILED - {results['error']}")
            continue
            
        config = results["config"]
        print(f"\nBenchmark {i+1}: {config.env_id}")
        print(f"  Config: {config.n_envs} envs, {config.n_steps} steps, {config.hidden_dims}")
        print(f"  Time: {results['mean_time']:.4f} ± {results['std_time']:.4f}s")
        print(f"  Throughput: {results['steps_per_second']:.1f} steps/sec, {results['episodes_per_second']:.1f} episodes/sec")
    
    return all_results


def main():
    """Main function for running benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark collect_rollouts function")
    parser.add_argument("--env", default="CartPole-v1", help="Environment ID")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=1024, help="Number of steps per rollout")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 64], help="Hidden layer dimensions")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--suite", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--collect-frames", action="store_true", help="Collect environment frames")
    
    args = parser.parse_args()
    
    if args.suite:
        run_benchmark_suite()
    else:
        # Single benchmark run
        device = torch.device("cuda" if torch.cuda.is_available() 
                            else "mps" if torch.backends.mps.is_available() 
                            else "cpu")
        
        config = BenchmarkConfig(
            env_id=args.env,
            n_envs=args.n_envs,
            n_steps=args.n_steps,
            hidden_dims=args.hidden_dims,
            deterministic=args.deterministic,
            collect_frames=args.collect_frames
        )
        
        benchmark_collect_rollouts(config, device, args.runs)


if __name__ == "__main__":
    main()
