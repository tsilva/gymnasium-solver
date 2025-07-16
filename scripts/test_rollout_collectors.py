import argparse
import time
import numpy as np
import torch
import gymnasium as gym
from utils.rollouts import SyncRolloutCollector, AsyncRolloutCollector


class SimpleConfig:
    def __init__(
        self, seed: int, train_rollout_steps: int, async_rollouts: bool, n_envs: int
    ):
        self.seed = seed
        self.train_rollout_steps = train_rollout_steps
        self.async_rollouts = async_rollouts
        self.n_envs = n_envs


def build_env(env_id: str, seed: int, n_envs: int):
    env = gym.vector.make(env_id, num_envs=n_envs, asynchronous=False)
    env.reset(seed=seed)
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
    args = parser.parse_args()

    config = SimpleConfig(
        seed=args.seed,
        train_rollout_steps=args.rollout_steps,
        async_rollouts=(args.collector == "async"),
        n_envs=args.n_envs,
    )

    env = build_env(args.env, args.seed, args.n_envs)
    policy_model, value_model = make_models(
        env.single_observation_space, env.single_action_space
    )

    collector_cls = (
        AsyncRolloutCollector if args.collector == "async" else SyncRolloutCollector
    )
    collector = collector_cls(config, env, policy_model, value_model)
    collector.start()

    print(
        f"Running {args.collector} rollout collector on {args.env}. Press Ctrl+C to stop."
    )
    start_time = time.time()
    total_steps = 0
    rollout_count = 0
    try:
        while True:
            roll_start = time.time()
            trajectories = collector.get_rollout(timeout=10.0)
            if trajectories is None:
                continue
            roll_time = time.time() - roll_start
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


if __name__ == "__main__":
    main()
