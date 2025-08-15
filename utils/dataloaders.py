from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from .datasets import IndexDataset
from .samplers import MultiPassRandomSampler


def build_dummy_loader(*, n_samples: int = 1, sample_dim: int = 1, batch_size: int = 1, num_workers: int = 0) -> DataLoader:
    """Return a trivial DataLoader that satisfies frameworks expecting one.

    Useful when the validation loop doesn't consume DataLoader data and drives
    evaluation procedurally (e.g., via environment rollouts inside validation_step).
    """
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
    """Efficient DataLoader for trajectory tensors using index-collate.

    It wraps the dataset length with an index-only Dataset and uses a custom
    collate_fn that slices all rollout tensors once per batch via the provided
    collector.

    Parameters
    - collector: RolloutCollector with `slice_trajectories(traj, idxs)`
    - trajectories: collected rollout data to be sliced by indices
    - num_passes: number of full passes (independent permutations) per epoch
    """
    # Resolve trajectories (if a getter is provided, prefer dynamic fetch)
    _traj = trajectories_getter() if trajectories_getter is not None else trajectories
    if _traj is None:
        raise ValueError("Either 'trajectories' or 'trajectories_getter' must be provided")

    # Determine dataset length from trajectories (observations is authoritative)
    data_len = len(_traj.observations)

    # Always use a MultiPassRandomSampler for index generation
    _sampler = MultiPassRandomSampler(data_len=data_len, num_passes=num_passes, generator=generator)

    # Index-only dataset; collate slices tensors once per batch
    index_ds = IndexDataset(data_len)
    def collate_fn(idxs):
        traj = trajectories_getter() if trajectories_getter is not None else _traj
        return collector.slice_trajectories(traj, idxs)

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
