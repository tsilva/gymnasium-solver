"""Benchmark Stable-Baselines3 rollout collection throughput.

This measures steps-per-second for SB3's PPO rollout collection, using
hyperparameters that match the CartPole-v1_ppo config in this repo by default.

Examples:
  python scripts/benchmark_sb3_rollout_collector.py --rollouts 50 --print_each
  python scripts/benchmark_sb3_rollout_collector.py --duration 10
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import sys as _sys

# Ensure project root on path to reuse config loader
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from utils.config import load_config

# Prefer locally cloned stable-baselines3 if present
_LOCAL_SB3 = _ROOT / "stable-baselines3"
if _LOCAL_SB3.exists() and str(_LOCAL_SB3) not in _sys.path:
    _sys.path.insert(0, str(_LOCAL_SB3))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn


class _NullCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark SB3 PPO rollout throughput")
    p.add_argument("--config", type=str, default="CartPole-v1_ppo", help="Config ID to match (env + hyperparams)")
    p.add_argument("--rollouts", type=int, default=50, help="Number of rollouts to collect (ignored if --duration)")
    p.add_argument("--duration", type=float, default=None, help="Duration in seconds to run instead of fixed rollouts")
    p.add_argument("--warmup", type=int, default=3, help="Warmup rollouts before measuring")
    p.add_argument("--print_each", action="store_true", help="Print per-rollout FPS")
    p.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv instead of DummyVecEnv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    print("=== SB3 Rollout Throughput Benchmark ===")
    print(f"Config: {args.config} | Env: {cfg.env_id}")
    print(f"n_envs={cfg.n_envs} n_steps={cfg.n_steps} gamma={cfg.gamma} gae_lambda={cfg.gae_lambda} lr={cfg.policy_lr}")

    # Build env similar to our project defaults (no recording/normalize by default for CartPole config)
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec_kwargs = {"start_method": "spawn"} if args.subproc else {}
    vec_env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, vec_env_cls=vec_cls, vec_env_kwargs=vec_kwargs)

    # Build SB3 PPO with matching hyperparams
    policy_kwargs = dict(net_arch=[cfg.hidden_dims if isinstance(cfg.hidden_dims, int) else list(cfg.hidden_dims)], activation_fn=nn.ReLU)
    # Flatten potential nested list from previous line (net_arch expects list of ints or dict for pi/vf)
    net_arch = policy_kwargs["net_arch"][0]
    if isinstance(net_arch, list):
        policy_kwargs["net_arch"] = net_arch

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        learning_rate=cfg.policy_lr,
        clip_range=cfg.clip_range,
        n_epochs=cfg.n_epochs,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    # Initialize learn state so collect_rollouts can be used directly
    model._setup_learn(total_timesteps=1)
    cb = _NullCallback()
    cb.init_callback(model)

    # Warmup
    for _ in range(max(0, args.warmup)):
        model.rollout_buffer.reset()
        model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=cfg.n_steps)

    per_fps = []
    per_steps = []
    start = time.perf_counter()

    if args.duration is not None and args.duration > 0:
        end_time = start + args.duration
        i = 0
        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            model.rollout_buffer.reset()
            model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=cfg.n_steps)
            t1 = time.perf_counter()
            steps = cfg.n_envs * cfg.n_steps
            fps = steps / max(1e-9, (t1 - t0))
            per_fps.append(fps)
            per_steps.append(steps)
            if args.print_each:
                print(f"rollout={i:04d} steps={steps} fps={fps:,.0f}")
            i += 1
    else:
        for i in range(max(1, args.rollouts)):
            t0 = time.perf_counter()
            model.rollout_buffer.reset()
            model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=cfg.n_steps)
            t1 = time.perf_counter()
            steps = cfg.n_envs * cfg.n_steps
            fps = steps / max(1e-9, (t1 - t0))
            per_fps.append(fps)
            per_steps.append(steps)
            if args.print_each:
                print(f"rollout={i:04d} steps={steps} fps={fps:,.0f}")

    elapsed = max(1e-9, time.perf_counter() - start)
    total_steps = int(np.sum(per_steps))
    sustained_fps = total_steps / elapsed

    print("\n=== Results (SB3) ===")
    if per_fps:
        print(f"Per-rollout FPS: mean={np.mean(per_fps):,.0f} median={np.median(per_fps):,.0f} min={np.min(per_fps):,.0f} max={np.max(per_fps):,.0f}")
    print(f"Elapsed: {elapsed:.2f}s | Total steps: {total_steps:,} | Sustained FPS: {sustained_fps:,.0f}")


if __name__ == "__main__":
    main()
