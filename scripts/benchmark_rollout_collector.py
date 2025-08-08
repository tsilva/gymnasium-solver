"""Benchmark RolloutCollector throughput.

This script measures the steps-per-second (FPS) achieved by the RolloutCollector
using hyperparameters from a given config (default: CartPole-v1_ppo).

Usage examples:
  - Benchmark for a fixed number of rollouts:
      python scripts/benchmark_rollout_collector.py --config CartPole-v1_ppo --rollouts 50

  - Benchmark for a fixed duration (seconds):
      python scripts/benchmark_rollout_collector.py --config CartPole-v1_ppo --duration 10
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is on sys.path when running as a script
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from utils.config import load_config
from utils.environment import build_env
from utils.models import ActorCritic
from utils.rollouts import RolloutCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RolloutCollector throughput")
    parser.add_argument("--config", type=str, default="CartPole-v1_ppo", help="Config ID to load")
    parser.add_argument("--rollouts", type=int, default=50, help="Number of rollouts to collect (ignored if --duration is set)")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds to run the benchmark (overrides --rollouts)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup rollouts to run before measuring")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions during collection")
    parser.add_argument("--print_each", action="store_true", help="Print per-rollout FPS")
    return parser.parse_args()


def build_policy_model(env, hidden_dims) -> ActorCritic:
    input_dim = env.get_input_dim()
    output_dim = env.get_output_dim()
    if input_dim is None or output_dim is None:
        raise RuntimeError("Could not infer model input/output dims from environment")
    return ActorCritic(input_dim, output_dim, hidden=hidden_dims if isinstance(hidden_dims, (list, tuple)) else (hidden_dims,))


def main() -> None:
    args = parse_args()

    # Load configuration (supports new environment-centric configs)
    config = load_config(args.config)

    # Report key settings
    print("=== RolloutCollector Throughput Benchmark ===")
    print(f"Config: {args.config}")
    print(f"Env: {config.env_id} | Algo: {config.algo_id}")
    print(f"n_envs={config.n_envs} | n_steps={config.n_steps} | batch_size={config.batch_size} | n_epochs={config.n_epochs}")
    print(f"gamma={config.gamma} | gae_lambda={config.gae_lambda} | clip_range={config.clip_range} | ent_coef={config.ent_coef}")

    # Build training environment (mirrors BaseAgent)
    env = build_env(
        config.env_id,
        seed=config.seed,
        n_envs=config.n_envs,
        subproc=config.subproc,
        obs_type=config.obs_type,
        env_wrappers=config.env_wrappers,
        norm_obs=config.normalize_obs,
        frame_stack=config.frame_stack,
        render_mode=None,
        env_kwargs=config.env_kwargs,
    )

    # Build policy model matching PPO config
    policy_model = build_policy_model(env, config.hidden_dims)

    # Build rollout collector using config-provided hyperparams
    collector = RolloutCollector(
        env,
        policy_model,
        n_steps=config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    # Warmup
    for i in range(max(0, args.warmup)):
        collector.collect(deterministic=args.deterministic)

    # Benchmark loop
    per_collect_fps = []
    per_collect_steps = []
    start = time.perf_counter()

    if args.duration is not None and args.duration > 0:
        end_time = start + args.duration
        i = 0
        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            collector.collect(deterministic=args.deterministic)
            t1 = time.perf_counter()
            steps = collector.rollout_steps
            fps = steps / max(1e-9, (t1 - t0))
            per_collect_fps.append(fps)
            per_collect_steps.append(steps)
            if args.print_each:
                print(f"rollout={i:04d} steps={steps} fps={fps:,.0f}")
            i += 1
    else:
        total_rollouts = max(1, args.rollouts)
        for i in range(total_rollouts):
            t0 = time.perf_counter()
            collector.collect(deterministic=args.deterministic)
            t1 = time.perf_counter()
            steps = collector.rollout_steps
            fps = steps / max(1e-9, (t1 - t0))
            per_collect_fps.append(fps)
            per_collect_steps.append(steps)
            if args.print_each:
                print(f"rollout={i:04d} steps={steps} fps={fps:,.0f}")

    elapsed = max(1e-9, time.perf_counter() - start)
    total_steps = int(np.sum(per_collect_steps))
    sustained_fps = total_steps / elapsed

    # Collector-internal stats (rolling window mean)
    metrics = collector.get_metrics()
    window_fps = metrics.get("rollout_fps", 0.0)

    print("\n=== Results ===")
    print(f"Elapsed: {elapsed:.2f}s | Total steps: {total_steps:,} | Sustained FPS: {sustained_fps:,.0f}")
    if per_collect_fps:
        print(f"Per-rollout FPS -> mean: {np.mean(per_collect_fps):,.0f}, median: {np.median(per_collect_fps):,.0f}, min: {np.min(per_collect_fps):,.0f}, max: {np.max(per_collect_fps):,.0f}")
    print(f"Collector window FPS (mean over last {collector.stats_window_size}): {window_fps:,.0f}")


if __name__ == "__main__":
    main()
