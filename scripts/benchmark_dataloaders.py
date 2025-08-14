"""
Benchmark DataLoader configurations on a real CartPole-v1 rollout:

1) Regular DataLoader with shuffle=True (recreate per pass)
2) Regular DataLoader with shuffle=False (recreate per pass)
3) DataLoader with MultiPassRandomSampler (shuffle=False; single loader with num_passes)

We first load the CartPole-v1_ppo config, build the vec env, create an ActorCritic
policy with matching hidden_dims, collect a single rollout using RolloutCollector,
then benchmark iterating that dataset with the three strategies.

Usage:
    python scripts/benchmark_dataloaders.py --config-id CartPole-v1_ppo --num-workers 2 --passes 20

Notes:
- Each rollout produces n_envs * n_steps samples (e.g., 8*32=256 for CartPole-v1_ppo).
- Set --num-workers to your CPU. Persistent workers are enabled by default.
- Use --pin-memory to enable pinned memory.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.samplers import MultiPassRandomSampler
from utils.environment import build_env
from utils.config import load_config
from utils.models import ActorCritic


@dataclass
class BenchmarkConfig:
    config_id: str = "CartPole-v1_ppo"
    batch_size: int | None = None  # if None, take from loaded config
    num_workers: int = 4
    passes: int | None = None      # if None, take from loaded config (n_epochs)
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int | None = None  # None => use PyTorch default


def make_rollout_dataset(cfg: BenchmarkConfig):
    # Load experiment config and env
    exp_cfg = load_config(cfg.config_id)
    env = build_env(
        exp_cfg.env_id,
        seed=exp_cfg.seed,
        n_envs=exp_cfg.n_envs,
        subproc=exp_cfg.subproc,
        obs_type=exp_cfg.obs_type,
        env_wrappers=exp_cfg.env_wrappers,
        norm_obs=exp_cfg.normalize_obs,
        frame_stack=exp_cfg.frame_stack,
        render_mode=None,
        env_kwargs=exp_cfg.env_kwargs,
    )

    # Build policy model compatible with env
    input_dim = env.get_input_dim()
    output_dim = env.get_output_dim()
    policy = ActorCritic(input_dim, output_dim, hidden=exp_cfg.hidden_dims).to(torch.device("cpu"))

    # Build collector and collect one rollout
    from utils.rollouts import RolloutCollector
    collector = RolloutCollector(
        env,
        policy,
        n_steps=exp_cfg.n_steps,
        **exp_cfg.rollout_collector_hyperparams(),
    )
    collector.collect(deterministic=False)

    # Decide passes and batch size from config if not provided
    passes = cfg.passes if cfg.passes is not None else exp_cfg.n_epochs
    batch_size = cfg.batch_size if cfg.batch_size is not None else exp_cfg.batch_size

    return env, collector.dataset, passes, batch_size


def count_batch_samples(batch: Tuple[torch.Tensor, ...]) -> int:
    # Assumes first tensor dimension corresponds to batch size
    return batch[0].shape[0]


def iterate_loader(loader: DataLoader) -> Tuple[int, int]:
    """Iterate through loader once, returning (num_batches, num_samples)."""
    num_batches = 0
    num_samples = 0
    for batch in loader:
        num_batches += 1
        num_samples += count_batch_samples(batch)
    return num_batches, num_samples


def build_loader(dataset, *, batch_size: int, cfg: BenchmarkConfig, shuffle: bool = False, sampler=None) -> DataLoader:
    kwargs: Dict[str, Any] = dict(
    dataset=dataset,
    batch_size=batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        drop_last=False,
    )
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def time_it(fn):
    t0 = time.perf_counter()
    result = fn()
    t1 = time.perf_counter()
    return result, t1 - t0


def run_benchmark(cfg: BenchmarkConfig) -> Dict[str, Any]:
    env, dataset, passes, batch_size = make_rollout_dataset(cfg)

    # 1) Shuffle=True, recreate DataLoader per epoch
    def run_shuffle_true():
        total_batches, total_samples = 0, 0
        loader = build_loader(dataset, batch_size=batch_size, cfg=cfg, shuffle=True)
        for _ in range(passes):
            b, s = iterate_loader(loader)
            total_batches += b
            total_samples += s
        return total_batches, total_samples

    # 2) Shuffle=False, recreate DataLoader per epoch
    def run_shuffle_false():
        total_batches, total_samples = 0, 0
        loader = build_loader(dataset, batch_size=batch_size, cfg=cfg, shuffle=False)
        for _ in range(passes):
            b, s = iterate_loader(loader)
            total_batches += b
            total_samples += s
        return total_batches, total_samples

    # 3) MultiPassRandomSampler, single DataLoader, single pass over loader
    def run_multipass_sampler():
        sampler = MultiPassRandomSampler(data_len=len(dataset), num_passes=passes)
        loader = build_loader(dataset, batch_size=batch_size, cfg=cfg, shuffle=False, sampler=sampler)
        return iterate_loader(loader)

    # Warmup: a short pass to spawn workers and stabilize timings
    warmup_loader = build_loader(dataset, batch_size=batch_size, cfg=cfg, shuffle=False)
    for _ in range(min(5, len(warmup_loader))):
        next(iter(warmup_loader))

    (shuffle_true_stats, shuffle_true_time) = time_it(run_shuffle_true)
    (shuffle_false_stats, shuffle_false_time) = time_it(run_shuffle_false)
    (multipass_stats, multipass_time) = time_it(run_multipass_sampler)

    keys = [
        ("shuffle_true", shuffle_true_stats, shuffle_true_time),
        ("shuffle_false", shuffle_false_stats, shuffle_false_time),
        ("multipass_sampler", multipass_stats, multipass_time),
    ]

    results: Dict[str, Any] = {
        "config": vars(cfg),
        "results": {}
    }

    for name, (batches, samples), seconds in keys:
        results["results"][name] = {
            "seconds": seconds,
            "batches": batches,
            "samples": samples,
            "batches_per_sec": batches / seconds if seconds > 0 else float("inf"),
            "samples_per_sec": samples / seconds if seconds > 0 else float("inf"),
        }

    try:
        env.close()
    except Exception:
        pass

    return results


def parse_args() -> BenchmarkConfig:
    p = argparse.ArgumentParser(description="Benchmark DataLoader strategies on a real CartPole-v1 rollout")
    p.add_argument("--config-id", type=str, default="CartPole-v1_ppo")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size (defaults to config.batch_size)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--passes", type=int, default=None, help="Override number of passes (defaults to config.n_epochs)")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-persistent-workers", action="store_true")
    p.add_argument("--prefetch-factor", type=int, default=None)
    args = p.parse_args()
    return BenchmarkConfig(
        config_id=args.config_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        passes=args.passes,
        pin_memory=args.pin_memory,
        persistent_workers=not args.no_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )


def main():
    cfg = parse_args()
    results = run_benchmark(cfg)
    # Pretty print table-like summary
    print("\n=== DataLoader Benchmark Results ===")
    print(f"config_id={cfg.config_id}, batch_size={cfg.batch_size}, num_workers={cfg.num_workers}, passes={cfg.passes}, "
          f"pin_memory={cfg.pin_memory}, persistent_workers={cfg.persistent_workers}, prefetch_factor={cfg.prefetch_factor}")
    for name, stats in results["results"].items():
        print(f"- {name:17s} | time={stats['seconds']:.3f}s | batches={stats['batches']:,} | "
              f"samples={stats['samples']:,} | bps={stats['batches_per_sec']:.1f} | sps={stats['samples_per_sec']:.0f}")
    # Also emit machine-readable JSON for downstream analysis
    print("\nJSON:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
