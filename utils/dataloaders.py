from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .datasets import IndexDataset
from .samplers import MultiPassRandomSampler


def build_dummy_loader(*, n_samples: int = 1, sample_dim: int = 1, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
    """Return a trivial DataLoader for frameworks that expect one."""
    dummy_data = torch.zeros(n_samples, sample_dim)
    dummy_target = torch.zeros(n_samples, sample_dim)
    dataset = TensorDataset(dummy_data, dummy_target)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def build_index_collate_loader_from_collector(
    *,
    collector,
    trajectories: Optional[Any] = None,
    trajectories_getter: Optional[Callable[[], Any]] = None,
    batch_size: int,
    num_passes: int,
    generator: Optional[torch.Generator] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    """Efficient DataLoader for trajectory tensors using index-collate."""
    # Resolve trajectories (if a getter is provided, prefer dynamic fetch)
    _traj = trajectories_getter() if trajectories_getter is not None else trajectories
    if _traj is None:
        raise ValueError("Either 'trajectories' or 'trajectories_getter' must be provided")

    # Determine dataset length from trajectories (observations is authoritative)
    data_len = len(_traj.observations)

    # Assert uniform batch sizes: the last batch must not be smaller.
    # This ensures consistent shapes and avoids edge cases in training loops.
    if data_len % int(batch_size) != 0:
        raise ValueError(
            f"Batch size must divide rollout size exactly: data_len={data_len}, batch_size={batch_size}. "
            "Choose a batch_size (or fraction of n_envs*n_steps) that evenly divides the rollout."
        )

    # Always use a MultiPassRandomSampler for index generation
    _sampler = MultiPassRandomSampler(data_len=data_len, num_passes=num_passes, generator=generator)

    # Index-only dataset; collate slices tensors once per batch
    index_ds = IndexDataset(data_len)
    def collate_fn(idxs):
        traj = trajectories_getter() if trajectories_getter is not None else _traj
        return collector.slice_trajectories(traj, idxs)

    # NOTE: num_workers=0 is intentional for this training scheme.
    # We collect a new rollout synchronously each epoch and keep it in memory.
    # The DataLoader simply indexes pre-collected tensors; multi-process workers
    # would add IPC serialization overhead without benefit.
    kwargs = dict(
        dataset=index_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=_sampler,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)
