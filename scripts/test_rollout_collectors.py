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


def main():
    parser = argparse.ArgumentParser(description="Test rollout collectors")
    parser.add_argument("--env", required=True, help="Gymnasium environment id")
    parser.add_argument("--collector", choices=["sync", "async"], default="async")
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-rollouts", type=int, default=None, help="Number of rollouts to collect (default: infinite)")
    args = parser.parse_args()

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
