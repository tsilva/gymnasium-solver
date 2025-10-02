from typing import NamedTuple, Tuple

import numpy as np
import torch


def _to_np(t: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return t.detach().cpu().numpy().astype(dtype, copy=False)


def _flat_env_major(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    """Flatten a (T,N,...) slice [start:end) into a (N*T,) env-major 1D array."""
    return arr[start:end].transpose(1, 0).reshape(-1)


class RolloutTrajectory(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor


class RolloutBuffer:
    """Persistent CPU rollout storage with preallocated (maxsize, n_envs, ...) buffers."""

    def __init__(
        self,
        n_envs: int,
        obs_shape: Tuple[int, ...],
        obs_dtype: np.dtype,
        device: torch.device,
        maxsize: int
    ) -> None:
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.device = device
        self.maxsize = int(maxsize)

        # Initialize buffers
        self.obs_buf = np.zeros((self.maxsize, self.n_envs, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obs_buf = np.zeros((self.maxsize, self.n_envs, *self.obs_shape), dtype=self.obs_dtype)
        self.actions_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.int64)
        self.rewards_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)
        self.values_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)
        self.dones_buf = np.zeros((self.maxsize, self.n_envs), dtype=bool)
        self.logprobs_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)
        self.timeouts_buf = np.zeros((self.maxsize, self.n_envs), dtype=bool)
        self.bootstrapped_values_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)

        # TODO: review this
        # write position tracking
        self.pos = 0
        self.size = 0  # number of valid steps currently stored (for info)

    def begin_rollout(self, T: int) -> int:
        """Ensure contiguous space for T steps; wrap to 0 if needed and return start index."""
        # In case the requested rollout size exceeds the buffer maxsize, raise an error
        # (otherwise data from the requested rollout will be overwritten by the rollout itself)
        if T > self.maxsize: raise ValueError(f"Rollout length T={T} exceeds buffer maxsize={self.maxsize}")

        # TODO: is this correct?
        # If the position plus the requested rollout size exceeds
        # the buffer maxsize, wrap around to the beginning
        if self.pos + T > self.maxsize: self.pos = 0

        # TODO: not sure I understand this
        start = self.pos
        self.pos += T
        self.size = max(self.size, self.pos)
        return start

    def add(
        self,
        idx: int,
        obs_np: np.ndarray,
        next_obs_np: np.ndarray,
        actions_np: np.ndarray,
        logps_np: np.ndarray,
        values_np: np.ndarray,
        rewards_np: np.ndarray,
        dones_np: np.ndarray,
        timeouts_np: np.ndarray,
    ) -> None:
        assert obs_np.shape == (self.n_envs, *self.obs_shape), f"Expected shape {(self.n_envs, *self.obs_shape)}, got {obs_np.shape}"
        self.obs_buf[idx] = obs_np
        self.next_obs_buf[idx] = next_obs_np
        self.actions_buf[idx] = actions_np
        self.logprobs_buf[idx] = logps_np
        self.values_buf[idx] = values_np
        self.rewards_buf[idx] = rewards_np
        self.dones_buf[idx] = dones_np
        self.timeouts_buf[idx] = timeouts_np

    # -------- Flatten helpers --------
    def flatten_slice_env_major(
        self,
        start: int,
        end: int,
        advantages_buf: np.ndarray,
        returns_buf: np.ndarray,
    ) -> RolloutTrajectory:
        """Create torch training tensors for the contiguous slice [start, end)."""
        T = end - start
        n_envs = self.n_envs

        # Observations: use CPU buffers and convert at return time
        obs_np = self.obs_buf[start:end]  # (T, N, *obs_shape)
        obs_t = torch.as_tensor(obs_np, device=self.device)
        obs_t_env_major = obs_t.transpose(0, 1)  # (N, T, *obs_shape)
        # Preserve image tensors (C,H,W) without flattening; flatten only vectors/scalars
        if len(self.obs_shape) == 0:
            observations = obs_t_env_major.reshape(n_envs * T, 1)
        elif len(self.obs_shape) == 1:
            observations = obs_t_env_major.reshape(n_envs * T, int(self.obs_shape[0]))
        else:
            observations = obs_t_env_major.reshape(n_envs * T, *self.obs_shape)

        def _flat_env_major_cpu_to_torch(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(_flat_env_major(arr, start, end), dtype=dtype, device=self.device)

        # TODO: why env major, and can we do env major by default if necessary?
        actions = _flat_env_major_cpu_to_torch(self.actions_buf, torch.int64)
        logprobs = _flat_env_major_cpu_to_torch(self.logprobs_buf, torch.float32)
        values = _flat_env_major_cpu_to_torch(self.values_buf, torch.float32)

        rewards = _flat_env_major_cpu_to_torch(self.rewards_buf, torch.float32)
        dones = _flat_env_major_cpu_to_torch(self.dones_buf, torch.bool)
        advantages = _flat_env_major_cpu_to_torch(advantages_buf, torch.float32)
        returns = _flat_env_major_cpu_to_torch(returns_buf, torch.float32)

        # Next observations: same env-major flattening as observations
        next_obs_tensor = torch.as_tensor(self.next_obs_buf[start:end], device=self.device)
        next_obs_env_major = next_obs_tensor.transpose(0, 1)
        if len(self.obs_shape) == 0:
            next_observations = next_obs_env_major.reshape(n_envs * T, 1)
        elif len(self.obs_shape) == 1:
            next_observations = next_obs_env_major.reshape(n_envs * T, int(self.obs_shape[0]))
        else:
            next_observations = next_obs_env_major.reshape(n_envs * T, *self.obs_shape)

        return RolloutTrajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            logprobs=logprobs,
            values=values,
            advantages=advantages,
            returns=returns,
            next_observations=next_observations,
        )
