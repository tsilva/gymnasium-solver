from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# TODO: review and CLEANUP

@dataclass
class _ReplayStorage:
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor  # behavior policy log-probs at collection time
    advantages: torch.Tensor
    returns: torch.Tensor


class ReplayBuffer:
    """Simple ring replay buffer for flattened transition batches.

    Stores per-timestep tensors already flattened env-major, matching
    RolloutTrajectory fields used by PPO: observations, actions,
    old logprobs, advantages, and returns. Values/dones are not
    required for the loss and are omitted for simplicity.
    """

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be > 0")
        self.capacity = int(capacity)
        self._storage: Optional[_ReplayStorage] = None
        self._pos = 0
        self._size = 0

    @property
    def size(self) -> int:
        return int(self._size)

    def _ensure_storage(self, traj) -> None:
        if self._storage is not None:
            return
        # Allocate CPU tensors with the right shapes/dtypes; keep CPU to be lightweight
        obs = traj.observations.detach().to("cpu")
        actions = traj.actions.detach().to(torch.long, copy=False).to("cpu")
        logprobs = traj.logprobs.detach().to(torch.float32, copy=False).to("cpu")
        adv = traj.advantages.detach().to(torch.float32, copy=False).to("cpu")
        rets = traj.returns.detach().to(torch.float32, copy=False).to("cpu")

        obs_shape = (self.capacity, *obs.shape[1:])
        self._storage = _ReplayStorage(
            observations=torch.zeros(obs_shape, dtype=obs.dtype),
            actions=torch.zeros((self.capacity,), dtype=actions.dtype),
            logprobs=torch.zeros((self.capacity,), dtype=logprobs.dtype),
            advantages=torch.zeros((self.capacity,), dtype=adv.dtype),
            returns=torch.zeros((self.capacity,), dtype=rets.dtype),
        )

    def add_trajectories(self, traj) -> None:
        """Append a flattened trajectory batch into the ring buffer."""
        if traj is None:
            return
        self._ensure_storage(traj)
        assert self._storage is not None

        # Prepare source tensors (CPU, detached)
        obs = traj.observations.detach().to("cpu")
        actions = traj.actions.detach().to(torch.long, copy=False).to("cpu")
        logprobs = traj.logprobs.detach().to(torch.float32, copy=False).to("cpu")
        adv = traj.advantages.detach().to(torch.float32, copy=False).to("cpu")
        rets = traj.returns.detach().to(torch.float32, copy=False).to("cpu")

        n = obs.shape[0]
        # Ring write with potential wrap
        end = self._pos + n
        if end <= self.capacity:
            sl = slice(self._pos, end)
            self._storage.observations[sl] = obs
            self._storage.actions[sl] = actions
            self._storage.logprobs[sl] = logprobs
            self._storage.advantages[sl] = adv
            self._storage.returns[sl] = rets
        else:
            first = self.capacity - self._pos
            sl1 = slice(self._pos, self.capacity)
            sl2 = slice(0, end - self.capacity)
            self._storage.observations[sl1] = obs[:first]
            self._storage.actions[sl1] = actions[:first]
            self._storage.logprobs[sl1] = logprobs[:first]
            self._storage.advantages[sl1] = adv[:first]
            self._storage.returns[sl1] = rets[:first]
            self._storage.observations[sl2] = obs[first:]
            self._storage.actions[sl2] = actions[first:]
            self._storage.logprobs[sl2] = logprobs[first:]
            self._storage.advantages[sl2] = adv[first:]
            self._storage.returns[sl2] = rets[first:]

        self._pos = end % self.capacity
        self._size = min(self.capacity, self._size + n)

    def sample(self, batch_size: int, *, device: Optional[torch.device] = None):
        """Uniform sample without replacement when possible, else with replacement.

        Returns a lightweight namespace-like object with the same
        field names used in PPOAgent.losses_for_batch.
        """
        if self._storage is None or self._size == 0:
            raise RuntimeError("ReplayBuffer is empty; add trajectories before sampling")
        bs = int(batch_size)
        n = int(self._size)
        device = device or torch.device("cpu")

        # Draw indices (no replacement if enough, else fallback)
        if bs <= n:
            idxs = torch.randperm(n)[:bs]
        else:
            idxs = torch.randint(0, n, (bs,))

        st = self._storage
        # Assemble views and move to the requested device lazily
        obs = st.observations[:n][idxs].to(device)
        actions = st.actions[:n][idxs].to(device)
        logprobs = st.logprobs[:n][idxs].to(device)
        adv = st.advantages[:n][idxs].to(device)
        rets = st.returns[:n][idxs].to(device)

        # Return a minimal object exposing the required attributes
        class _Batch:
            pass

        b = _Batch()
        b.observations = obs
        b.actions = actions
        b.logprobs = logprobs
        b.advantages = adv
        b.returns = rets
        return b

