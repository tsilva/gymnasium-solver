"""
Benchmark DataLoader configurations on a real CartPole-v1 rollout:

1) Regular DataLoader with shuffle=True (recreate per pass)
2) Regular DataLoader with shuffle=False (recreate per pass)
3) DataLoader with MultiPassRandomSampler (shuffle=False; single loader with num_passes)
4) PyTorch Lightning 1-epoch train loop using MultiPassRandomSampler (measure Lightning overhead)

We first load the CartPole-v1_ppo config, build the vec env, create an ActorCritic
policy with matching hidden_dims, collect a single rollout using RolloutCollector,
then benchmark iterating that dataset with the strategies above.

Usage:
    python scripts/benchmark_dataloaders.py --config-id CartPole-v1_ppo --num-workers 2 --passes 20

Notes:
- Each rollout produces n_envs * n_steps samples (e.g., 8*32=256 for CartPole-v1_ppo).
- Set --num-workers to your CPU. Persistent workers are enabled by default.
- Use --pin-memory to enable pinned memory.
- For overhead comparison, set --passes 20 to iterate each rollout 20 times in a single epoch.
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
import logging
import warnings
from pytorch_lightning import LightningModule, Trainer

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

    # 0) Plain Python batching over underlying tensors (no DataLoader)
    def run_plain_python_multipass():
        traj = dataset.trajectories  # RolloutTrajectory of tensors
        n = len(dataset)
        # Build a single index list representing `passes` random passes over the data
        # to mirror MultiPassRandomSampler behavior
        idx_list = []
        for _ in range(passes):
            perm = torch.randperm(n).tolist()
            idx_list.extend(perm)

        total_batches, total_samples = 0, 0
        # Manual batching
        for i in range(0, len(idx_list), batch_size):
            idxs = idx_list[i:i + batch_size]
            # Slice each field to form a batch tuple (match first-dim semantics)
            batch = (
                traj.observations[idxs],
                traj.actions[idxs],
                traj.rewards[idxs],
                traj.dones[idxs],
                traj.old_log_prob[idxs],
                traj.old_values[idxs],
                traj.advantages[idxs],
                traj.returns[idxs],
                traj.next_observations[idxs],
            )
            total_batches += 1
            total_samples += count_batch_samples(batch)
        return total_batches, total_samples

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

    # 4) Lightning 1-epoch loop with MultiPassRandomSampler (measure framework overhead)
    class NoOpLightningModule(LightningModule):
        def __init__(self, dataset, batch_size: int, cfg: BenchmarkConfig, passes: int):
            super().__init__()
            self.dataset = dataset
            self.batch_size = batch_size
            self.cfg = cfg
            self.passes = passes
            self.batches = 0
            self.samples = 0
            self.automatic_optimization = False  # measure dataloader+loop overhead only
            # Dummy parameter to attach an optimizer and avoid Trainer warnings
            self._noop = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        def train_dataloader(self):
            sampler = MultiPassRandomSampler(data_len=len(self.dataset), num_passes=self.passes)
            return build_loader(self.dataset, batch_size=self.batch_size, cfg=self.cfg, shuffle=False, sampler=sampler)

        def on_train_epoch_start(self):
            self._start = time.time_ns()

        def training_step(self, batch, batch_idx):
            # Count batches/samples; no backward/optimizer for pure loop overhead
            bsz = count_batch_samples(batch)
            self.batches += 1
            self.samples += bsz
            return None

        def on_train_epoch_end(self):
            elapsed = (time.time_ns() - self._start) / 1e6
            print(f"Lightning epoch {self.current_epoch} completed in {elapsed:.2f}ms")

        def configure_optimizers(self):
            # Provide a no-op optimizer to suppress "no optimizer" warnings
            return torch.optim.SGD([self._noop], lr=0.0)

    # Silence Lightning info logs to avoid console overhead in timing
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    try:
        import lightning as _lt  # PL 2.x unifies under lightning namespace
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    except Exception:
        pass
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")

    # Pre-build model and trainer so the timer measures the epoch loop, not setup
    _lt_model = NoOpLightningModule(dataset, batch_size, cfg, passes)
    _lt_trainer = Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
    )

    def run_lightning_multipass_epoch():
        _lt_trainer.fit(_lt_model)
        return _lt_model.batches, _lt_model.samples

    # Warmup: a short pass to spawn workers and stabilize timings
    warmup_loader = build_loader(dataset, batch_size=batch_size, cfg=cfg, shuffle=False)
    for _ in range(min(5, len(warmup_loader))):
        next(iter(warmup_loader))

    (shuffle_true_stats, shuffle_true_time) = time_it(run_shuffle_true)
    (shuffle_false_stats, shuffle_false_time) = time_it(run_shuffle_false)
    (plain_python_stats, plain_python_time) = time_it(run_plain_python_multipass)
    (multipass_stats, multipass_time) = time_it(run_multipass_sampler)
    (lightning_stats, lightning_time) = time_it(run_lightning_multipass_epoch)

    keys = [
        ("plain_python_multipass", plain_python_stats, plain_python_time),
        ("shuffle_true", shuffle_true_stats, shuffle_true_time),
        ("shuffle_false", shuffle_false_stats, shuffle_false_time),
        ("multipass_sampler", multipass_stats, multipass_time),
        ("lightning_multipass_epoch", lightning_stats, lightning_time),
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
    cfg.config_id = "CartPole-v1_ppo"  # Ensure we use a known config for benchmarking
    cfg.num_workers = 0
    cfg.passes = 20  # Set a fixed number of passes for benchmarking

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
