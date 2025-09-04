import pytest
import torch
from types import SimpleNamespace


@pytest.mark.unit
def test_dataloader_reshuffles_across_epochs_same_rollout():
    """
    Confirm that iterating the DataLoader twice (simulating two epochs) yields
    different index orders even when the underlying rollout data is unchanged.

    This mirrors BaseAgent behavior where the DataLoader is constructed once
    with a MultiPassRandomSampler and a persistent torch.Generator, and each
    epoch iterates the same loader again against freshly collected rollouts.
    Here we keep the rollout the same to isolate the shuffling behavior.
    """

    data_len = 64
    batch_size = 16
    num_passes = 1  # one pass per epoch

    # Trajectories stub with observations-length only (used to determine data_len)
    traj = SimpleNamespace(observations=[0] * data_len)

    # Collector stub that returns the batch indices so we can inspect order
    class _CollectorStub:
        def slice_trajectories(self, traj, idxs):  # noqa: ARG002
            return torch.as_tensor(list(idxs), dtype=torch.int64)

    from utils.dataloaders import build_index_collate_loader_from_collector

    # Use a dedicated generator to avoid coupling with global state
    gen = torch.Generator().manual_seed(123)

    loader = build_index_collate_loader_from_collector(
        collector=_CollectorStub(),
        trajectories=traj,
        batch_size=batch_size,
        num_passes=num_passes,
        generator=gen,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # Epoch 0: collect full order from batches
    order_epoch0 = []
    for batch in loader:
        order_epoch0.extend(batch.tolist())

    # Sanity: it must be a permutation of [0..data_len-1]
    assert len(order_epoch0) == data_len
    assert sorted(order_epoch0) == list(range(data_len))

    # Epoch 1: iterating the same loader again should reshuffle
    order_epoch1 = []
    for batch in loader:
        order_epoch1.extend(batch.tolist())

    assert len(order_epoch1) == data_len
    assert sorted(order_epoch1) == list(range(data_len))

    # Core guarantee: different order across epochs even with the same rollout
    assert order_epoch0 != order_epoch1

