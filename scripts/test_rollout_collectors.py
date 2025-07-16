import argparse
import time
import numpy as np
import torch
import gymnasium as gym
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.rollouts import SyncRolloutCollector, AsyncRolloutCollector
from tsilva_notebook_utils.gymnasium import build_env as _build_env


class SimpleConfig:
    def __init__(
        self, seed: int, train_rollout_steps: int, async_rollouts: bool, n_envs: int
    ):
        self.seed = seed
        self.train_rollout_steps = train_rollout_steps
        self.async_rollouts = async_rollouts
        self.n_envs = n_envs


def build_env(env_id: str, seed: int, n_envs: int):
    # Use the same environment building approach as the main code
    env = _build_env(env_id, norm_obs=False, n_envs=n_envs, seed=seed)
    return env


def make_models(obs_space, act_space):
    if len(obs_space.shape) != 1:
        raise NotImplementedError("Only flat observation spaces are supported")
    obs_dim = obs_space.shape[0]
    if not isinstance(act_space, gym.spaces.Discrete):
        raise NotImplementedError("Only discrete action spaces are supported")
    act_dim = act_space.n
    policy_model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, act_dim),
    )
    value_model = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    )
    return policy_model, value_model


def run_single_test(env_id, seed, collector_type, n_envs, rollout_steps, n_rollouts):
    """Run a single performance test and return results"""
    config = SimpleConfig(
        seed=seed,
        train_rollout_steps=rollout_steps,
        async_rollouts=(collector_type == "async"),
        n_envs=n_envs,
    )

    env = build_env(env_id, seed, n_envs)
    
    # Get observation and action spaces - handle both vector and single envs
    if hasattr(env, 'single_observation_space'):
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        obs_space = env.observation_space
        act_space = env.action_space
    
    policy_model, value_model = make_models(obs_space, act_space)

    collector_cls = AsyncRolloutCollector if collector_type == "async" else SyncRolloutCollector
    collector = collector_cls(config, env, policy_model, value_model)
    collector.start()

    start_time = time.time()
    total_steps = 0
    rollout_count = 0
    rollout_times = []
    
    try:
        while rollout_count < n_rollouts:
            roll_start = time.time()
            trajectories = collector.get_rollout(timeout=10.0)
            if trajectories is None:
                continue
            roll_time = time.time() - roll_start
            rollout_times.append(roll_time)
            rollout_count += 1
            steps = len(trajectories[0])
            total_steps += steps
    finally:
        collector.stop()
        env.close()
        
    # Calculate statistics
    elapsed_total = time.time() - start_time
    avg_rollout_time = np.mean(rollout_times)
    std_rollout_time = np.std(rollout_times)
    final_steps_per_sec = total_steps / max(elapsed_total, 1e-6)
    steps_per_rollout = total_steps // max(rollout_count, 1)
    throughput_per_env = final_steps_per_sec / n_envs
    
    return {
        'collector': collector_type,
        'n_envs': n_envs,
        'rollout_steps': rollout_steps,
        'steps_per_rollout': steps_per_rollout,
        'total_steps': total_steps,
        'total_rollouts': rollout_count,
        'total_time': elapsed_total,
        'avg_rollout_time': avg_rollout_time,
        'std_rollout_time': std_rollout_time,
        'total_throughput': final_steps_per_sec,
        'throughput_per_env': throughput_per_env
    }


def run_performance_tests(env_id, seed):
    """Run comprehensive performance tests and display results in a table"""
    print(f"🚀 Running comprehensive rollout collection performance tests for {env_id}")
    print("This may take a few minutes...\n")
    
    # Test configurations: (collector, n_envs, rollout_steps, n_rollouts)
    test_configs = [
        ("sync", 1, 100, 10),
        ("sync", 2, 100, 10),
        ("sync", 4, 100, 10),
        ("async", 1, 100, 10),
        ("async", 2, 100, 10),
        ("async", 4, 100, 10),
        ("async", 8, 100, 5),
        ("async", 4, 256, 5),  # Test different rollout sizes
    ]
    
    results = []
    
    for i, (collector, n_envs, rollout_steps, n_rollouts) in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] Testing {collector} collector with {n_envs} envs, {rollout_steps} steps...")
        
        try:
            result = run_single_test(env_id, seed, collector, n_envs, rollout_steps, n_rollouts)
            results.append(result)
            print(f"  ✓ {result['total_throughput']:.1f} steps/s")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Display results table
    print("\n" + "="*80)
    print("📊 ROLLOUT COLLECTION PERFORMANCE RESULTS")
    print("="*80)
    
    # Table header
    header = f"{'Collector':<10} {'Envs':<5} {'Steps/Roll':<10} {'Total Steps/s':<12} {'Per-Env Steps/s':<15} {'Avg Roll Time':<12}"
    print(header)
    print("-" * len(header))
    
    # Sort results by total throughput descending
    results.sort(key=lambda x: x['total_throughput'], reverse=True)
    
    # Table rows
    for result in results:
        row = (f"{result['collector']:<10} "
               f"{result['n_envs']:<5} "
               f"{result['steps_per_rollout']:<10} "
               f"{result['total_throughput']:<12.1f} "
               f"{result['throughput_per_env']:<15.1f} "
               f"{result['avg_rollout_time']:<12.3f}")
        print(row)
    
    print("\n" + "="*80)
    print("📈 KEY INSIGHTS")
    print("="*80)
    
    # Find best configurations
    best_total = max(results, key=lambda x: x['total_throughput'])
    best_per_env = max(results, key=lambda x: x['throughput_per_env'])
    sync_results = [r for r in results if r['collector'] == 'sync']
    async_results = [r for r in results if r['collector'] == 'async']
    
    print(f"🏆 Best total throughput: {best_total['collector']} with {best_total['n_envs']} envs ({best_total['total_throughput']:.1f} steps/s)")
    print(f"🎯 Best per-env efficiency: {best_per_env['collector']} with {best_per_env['n_envs']} envs ({best_per_env['throughput_per_env']:.1f} steps/s/env)")
    
    if sync_results and async_results:
        # Compare sync vs async at same env count
        sync_4_env = next((r for r in sync_results if r['n_envs'] == 4), None)
        async_4_env = next((r for r in async_results if r['n_envs'] == 4), None)
        
        if sync_4_env and async_4_env:
            improvement = ((async_4_env['total_throughput'] - sync_4_env['total_throughput']) / 
                          sync_4_env['total_throughput'] * 100)
            print(f"⚡ Async vs Sync (4 envs): {improvement:+.1f}% throughput improvement")
    
    # Scaling analysis
    env_counts = sorted(set(r['n_envs'] for r in results))
    if len(env_counts) > 1:
        print(f"📊 Scaling: {env_counts[0]} → {env_counts[-1]} envs")
        for collector_type in ['sync', 'async']:
            type_results = [r for r in results if r['collector'] == collector_type]
            if len(type_results) >= 2:
                type_results.sort(key=lambda x: x['n_envs'])
                first = type_results[0]
                last = type_results[-1]
                scaling_efficiency = (last['total_throughput'] / last['n_envs']) / (first['total_throughput'] / first['n_envs'])
                print(f"  {collector_type}: {scaling_efficiency:.2f}x per-env efficiency retention")


def main():
    parser = argparse.ArgumentParser(description="Test rollout collectors")
    parser.add_argument("--env", required=True, help="Gymnasium environment id")
    parser.add_argument("--collector", choices=["sync", "async"], default="async")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=None, help="Number of rollouts to collect (default: infinite)")
    parser.add_argument("--test-mode", action="store_true", help="Run comprehensive performance tests and output table")
    args = parser.parse_args()

    if args.test_mode:
        run_performance_tests(args.env, args.seed)
        return

    config = SimpleConfig(
        seed=args.seed,
        train_rollout_steps=args.rollout_steps,
        async_rollouts=(args.collector == "async"),
        n_envs=args.n_envs,
    )

    env = build_env(args.env, args.seed, args.n_envs)
    
    # Get observation and action spaces - handle both vector and single envs
    if hasattr(env, 'single_observation_space'):
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        obs_space = env.observation_space
        act_space = env.action_space
    
    policy_model, value_model = make_models(
        obs_space, act_space
    )

    collector_cls = (
        AsyncRolloutCollector if args.collector == "async" else SyncRolloutCollector
    )
    collector = collector_cls(config, env, policy_model, value_model)
    collector.start()

    if args.n_rollouts:
        print(
            f"Running {args.collector} rollout collector on {args.env} for {args.n_rollouts} rollouts."
        )
    else:
        print(
            f"Running {args.collector} rollout collector on {args.env}. Press Ctrl+C to stop."
        )
    
    start_time = time.time()
    total_steps = 0
    rollout_count = 0
    rollout_times = []
    
    try:
        while True:
            # Check if we've reached the target number of rollouts
            if args.n_rollouts and rollout_count >= args.n_rollouts:
                break
                
            roll_start = time.time()
            trajectories = collector.get_rollout(timeout=10.0)
            if trajectories is None:
                continue
            roll_time = time.time() - roll_start
            rollout_times.append(roll_time)
            rollout_count += 1
            steps = len(trajectories[0])
            total_steps += steps
            mean_reward = trajectories[2].mean().item()
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / max(elapsed, 1e-6)
            print(
                f"Rollout {rollout_count:04d} | steps: {steps} | mean_reward: {mean_reward:.2f} "
                f"| rollout_time: {roll_time:.2f}s | steps/s: {steps_per_sec:.1f}"
            )
    except KeyboardInterrupt:
        print("Stopping collector...")
    finally:
        collector.stop()
        env.close()
        
        # Print summary statistics
        if rollout_times:
            elapsed_total = time.time() - start_time
            avg_rollout_time = np.mean(rollout_times)
            std_rollout_time = np.std(rollout_times)
            final_steps_per_sec = total_steps / max(elapsed_total, 1e-6)
            
            print(f"\n=== Performance Summary ===")
            print(f"Total rollouts: {rollout_count}")
            print(f"Total steps: {total_steps}")
            print(f"Total time: {elapsed_total:.2f}s")
            print(f"Average rollout time: {avg_rollout_time:.3f}s ± {std_rollout_time:.3f}s")
            print(f"Final throughput: {final_steps_per_sec:.1f} steps/s")
            print(f"Environments: {args.n_envs}")
            print(f"Steps per rollout: {total_steps // max(rollout_count, 1)}")
            print(f"Throughput per env: {final_steps_per_sec / args.n_envs:.1f} steps/s/env")


if __name__ == "__main__":
    main()
