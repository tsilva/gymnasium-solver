"""Unified benchmark for custom RolloutCollector vs Stable-Baselines3 collector.

This script runs both implementations under the same config/env and compares
throughput (steps-per-second). It reuses common setup and timing logic to
avoid duplication.

Examples:
  - Fixed number of rollouts per implementation (default both):
      python scripts/benchmark_collectors.py --config CartPole-v1_ppo --rollouts 50

  - Fixed benchmark duration per implementation:
      python scripts/benchmark_collectors.py --config CartPole-v1_ppo --duration 10

  - Run only one implementation:
      python scripts/benchmark_collectors.py --only custom
      python scripts/benchmark_collectors.py --only sb3
"""
from __future__ import annotations

import argparse
import sys as _sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Ensure project root is on sys.path when running as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

# Prefer locally vendored Stable-Baselines3 if present (repo has a copy)
_LOCAL_SB3 = _ROOT / "stable-baselines3"
if _LOCAL_SB3.exists() and str(_LOCAL_SB3) not in _sys.path:
    _sys.path.insert(0, str(_LOCAL_SB3))

from utils.config import load_config
from utils.environment import build_env

# Custom collector imports
from utils.models import ActorCritic
from utils.rollouts import RolloutCollector

# SB3 imports (optional)
try:
    import torch.nn as nn  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    _HAS_SB3 = True
except Exception:
    PPO = None  # type: ignore
    BaseCallback = object  # type: ignore
    nn = None  # type: ignore
    _HAS_SB3 = False


class _NullCallback(BaseCallback):
    def _on_step(self) -> bool:  # type: ignore[override]
        return True


@dataclass
class BenchmarkResult:
    label: str
    per_rollout_fps: list[float]
    per_rollout_steps: list[int]
    elapsed: float
    total_steps: int
    sustained_fps: float
    extra: dict


def _bool_flag(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y", "t")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark custom vs SB3 rollout collectors")
    p.add_argument("--config", type=str, default="CartPole-v1_ppo", help="Config ID to load")
    p.add_argument("--rollouts", type=int, default=50, help="Number of rollouts to collect (ignored if --duration)")
    p.add_argument("--duration", type=float, default=None, help="Duration in seconds to run instead of fixed rollouts")
    p.add_argument("--warmup", type=int, default=3, help="Warmup rollouts before measuring")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions in custom collector (SB3 always stochastic)")
    p.add_argument("--print_each", action="store_true", help="Print per-rollout FPS during timing")
    p.add_argument("--subproc", type=_bool_flag, nargs="?", const=True, default=None, help="Override vectorization with subprocess (true/false); defaults to config if omitted")
    p.add_argument("--only", choices=["both", "custom", "sb3"], default="both", help="Which implementation(s) to run")
    return p.parse_args()


def _effective_subproc(cfg, override: Optional[bool]) -> Optional[bool]:
    return override if override is not None else getattr(cfg, "subproc", None)


def _build_policy_model(env, hidden_dims) -> ActorCritic:
    input_dim = env.get_input_dim()
    output_dim = env.get_output_dim()
    if input_dim is None or output_dim is None:
        raise RuntimeError("Could not infer model input/output dims from environment")
    hd = hidden_dims if isinstance(hidden_dims, (list, tuple)) else (hidden_dims,)
    return ActorCritic(input_dim, output_dim, hidden_dims=hd)


def _run_benchmark_loop(
    *,
    collect_once: Callable[[], int],
    rollouts: Optional[int],
    duration: Optional[float],
    warmup: int,
    print_each: bool,
) -> tuple[list[float], list[int], float]:
    # Warmup
    for _ in range(max(0, warmup)):
        collect_once()

    per_fps: list[float] = []
    per_steps: list[int] = []
    start = time.perf_counter()

    if duration is not None and duration > 0:
        end_time = start + duration
        i = 0
        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            steps = int(collect_once())
            t1 = time.perf_counter()
            fps = steps / max(1e-9, (t1 - t0))
            per_fps.append(fps)
            per_steps.append(steps)
            if print_each:
                print(f"rollout={i:04d} steps={steps} fps={fps:,.0f}")
            i += 1
    else:
        n = max(1, int(rollouts or 1))
        for i in range(n):
            t0 = time.perf_counter()
            steps = int(collect_once())
            t1 = time.perf_counter()
            fps = steps / max(1e-9, (t1 - t0))
            per_fps.append(fps)
            per_steps.append(steps)
            if print_each:
                print(f"rollout={i:04d} steps={steps} fps={fps:,.0f}")

    elapsed = max(1e-9, time.perf_counter() - start)
    return per_fps, per_steps, elapsed


def run_custom_collector(cfg, *, subproc_override: Optional[bool], deterministic: bool, rollouts: Optional[int], duration: Optional[float], warmup: int, print_each: bool) -> BenchmarkResult:
    print("\n=== Custom RolloutCollector ===")

    vec_env = build_env(
        cfg.env_id,
        seed=cfg.seed,
        n_envs=cfg.n_envs,
        subproc=_effective_subproc(cfg, subproc_override),
        obs_type=cfg.obs_type,
        env_wrappers=cfg.env_wrappers,
        norm_obs=cfg.normalize_obs,
        frame_stack=cfg.frame_stack,
        render_mode=None,
        env_kwargs=cfg.env_kwargs,
    )

    policy_model = _build_policy_model(vec_env, cfg.hidden_dims)

    collector = RolloutCollector(
        vec_env,
        policy_model,
        n_steps=cfg.n_steps,
        **cfg.rollout_collector_hyperparams(),
    )

    def _collect_once() -> int:
        collector.collect(deterministic=deterministic)
        return int(collector.rollout_steps)

    per_fps, per_steps, elapsed = _run_benchmark_loop(
        collect_once=_collect_once,
        rollouts=rollouts,
        duration=duration,
        warmup=warmup,
        print_each=print_each,
    )

    total_steps = int(np.sum(per_steps))
    sustained_fps = total_steps / max(1e-9, elapsed)

    metrics = collector.get_metrics()
    window_fps = float(metrics.get("rollout_fps", 0.0))

    print("Results: elapsed={:.2f}s total_steps={:,} sustained_fps={:,.0f} (window_fps≈{:,.0f})".format(
        elapsed, total_steps, sustained_fps, window_fps
    ))

    return BenchmarkResult(
        label="custom",
        per_rollout_fps=per_fps,
        per_rollout_steps=[int(s) for s in per_steps],
        elapsed=elapsed,
        total_steps=total_steps,
        sustained_fps=sustained_fps,
        extra={"window_fps": window_fps},
    )


def run_sb3_collector(cfg, *, subproc_override: Optional[bool], rollouts: Optional[int], duration: Optional[float], warmup: int, print_each: bool) -> BenchmarkResult:
    if not _HAS_SB3:
        raise RuntimeError("Stable-Baselines3 is not available. Ensure dependency is installed or vendored.")

    print("\n=== SB3 PPO collect_rollouts ===")

    vec_env = build_env(
        cfg.env_id,
        seed=cfg.seed,
        n_envs=cfg.n_envs,
        subproc=_effective_subproc(cfg, subproc_override),
        obs_type=cfg.obs_type,
        env_wrappers=cfg.env_wrappers,
        norm_obs=cfg.normalize_obs,
        frame_stack=cfg.frame_stack,
        render_mode=None,
        env_kwargs=cfg.env_kwargs,
    )

    # Prepare policy kwargs (match hidden_dims and activation)
    layers = list(cfg.hidden_dims) if isinstance(cfg.hidden_dims, (list, tuple)) else [int(cfg.hidden_dims)]
    policy_kwargs = dict(net_arch=layers)
    if nn is not None:
        policy_kwargs["activation_fn"] = nn.ReLU

    model = PPO(
        policy="mlp",
        env=vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        policy_lr=cfg.policy_lr,
        clip_range=cfg.clip_range,
        n_epochs=cfg.n_epochs,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    # Initialize for collect_rollouts
    model._setup_learn(total_timesteps=1)
    cb = _NullCallback()  # type: ignore[abstract]
    cb.init_callback(model)

    def _collect_once() -> int:
        model.rollout_buffer.reset()
        model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=cfg.n_steps)
        return int(cfg.n_envs * cfg.n_steps)

    per_fps, per_steps, elapsed = _run_benchmark_loop(
        collect_once=_collect_once,
        rollouts=rollouts,
        duration=duration,
        warmup=warmup,
        print_each=print_each,
    )

    total_steps = int(np.sum(per_steps))
    sustained_fps = total_steps / max(1e-9, elapsed)

    print("Results: elapsed={:.2f}s total_steps={:,} sustained_fps={:,.0f}".format(
        elapsed, total_steps, sustained_fps
    ))

    return BenchmarkResult(
        label="sb3",
        per_rollout_fps=per_fps,
        per_rollout_steps=[int(s) for s in per_steps],
        elapsed=elapsed,
        total_steps=total_steps,
        sustained_fps=sustained_fps,
        extra={},
    )


def _print_header(cfg) -> None:
    print("=== Rollout Collection Throughput Benchmark (Custom vs SB3) ===")
    print(f"Config: {cfg.config_id if hasattr(cfg, 'config_id') else ''} | Env: {cfg.env_id} | Algo: {getattr(cfg, 'algo_id', '')}")
    print(
        "n_envs={} | n_steps={} | batch_size={} | n_epochs={} | gamma={} | gae_lambda={} | clip_range={} | ent_coef={}".format(
            cfg.n_envs, cfg.n_steps, cfg.batch_size, cfg.n_epochs, cfg.gamma, cfg.gae_lambda, cfg.clip_range, cfg.ent_coef
        )
    )


def _compare_and_report(custom_res: Optional[BenchmarkResult], sb3_res: Optional[BenchmarkResult]) -> None:
    print("\n=== Comparison ===")
    if custom_res is None and sb3_res is None:
        print("No results to compare.")
        return

    if custom_res is not None:
        print("Custom: sustained_fps={:,.0f} (elapsed={:.2f}s, steps={:,})".format(
            custom_res.sustained_fps, custom_res.elapsed, custom_res.total_steps
        ))
    if sb3_res is not None:
        print("SB3:    sustained_fps={:,.0f} (elapsed={:.2f}s, steps={:,})".format(
            sb3_res.sustained_fps, sb3_res.elapsed, sb3_res.total_steps
        ))

    if custom_res is not None and sb3_res is not None:
        a = custom_res.sustained_fps
        b = sb3_res.sustained_fps
        if a == 0 and b == 0:
            print("Both sustained FPS are zero; cannot compare.")
            return
        if a > b:
            faster = "Custom"
            slower = "SB3"
            pct = (a - b) / max(1e-9, b) * 100.0
        else:
            faster = "SB3"
            slower = "Custom"
            pct = (b - a) / max(1e-9, a) * 100.0
        print(f"Winner: {faster} (~{pct:.1f}% faster than {slower})")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    _print_header(cfg)

    custom_res: Optional[BenchmarkResult] = None
    sb3_res: Optional[BenchmarkResult] = None

    if args.only in ("both", "custom"):
        custom_res = run_custom_collector(
            cfg,
            subproc_override=args.subproc,
            deterministic=args.deterministic,
            rollouts=args.rollouts,
            duration=args.duration,
            warmup=args.warmup,
            print_each=args.print_each,
        )

    if args.only in ("both", "sb3"):
        if not _HAS_SB3:
            print("SB3 not available; skipping SB3 benchmark.")
        else:
            sb3_res = run_sb3_collector(
                cfg,
                subproc_override=args.subproc,
                rollouts=args.rollouts,
                duration=args.duration,
                warmup=args.warmup,
                print_each=args.print_each,
            )

    _compare_and_report(custom_res, sb3_res)


if __name__ == "__main__":
    main()
