import time
from collections import deque
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

from utils.torch import _device_of, inference_ctx


# -----------------------------
# Internal small helpers (DRY)
# -----------------------------
def _canonicalize_timeouts(dones: np.ndarray, timeouts: Optional[np.ndarray]) -> np.ndarray:
    """Return a boolean timeouts array with same shape as dones.

    When ``timeouts`` is None, returns an all-False array. Ensures dtype bool.
    """
    if timeouts is None:
        return np.zeros_like(dones, dtype=bool)
    return np.asarray(timeouts, dtype=bool)


def _real_terminal_mask(dones: np.ndarray, timeouts: Optional[np.ndarray]) -> np.ndarray:
    """Real terminal if done and not a time-limit truncation."""
    dones_b = np.asarray(dones, dtype=bool)
    timeouts_b = _canonicalize_timeouts(dones_b, timeouts)
    return np.logical_and(dones_b, ~timeouts_b)


def _non_terminal_float_mask(dones: np.ndarray, timeouts: Optional[np.ndarray]) -> np.ndarray:
    """Float32 mask of non-terminals given dones/timeouts."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    return (~real_terminal).astype(np.float32)


def _build_idx_map_from_valid_mask(valid_mask_flat: np.ndarray) -> Optional[np.ndarray]:
    """Vectorized map from each position to the nearest previous valid index.

    Any positions before the first valid entry map to the first valid index to
    keep the dataset length stable. Returns None when there are no valid entries.
    """
    valid_mask_flat = np.asarray(valid_mask_flat, dtype=bool)
    if not valid_mask_flat.any():
        return None
    N = int(valid_mask_flat.size)
    idxs = np.arange(N, dtype=np.int64)
    cur = np.where(valid_mask_flat, idxs, -1)
    filled = np.maximum.accumulate(cur)
    first_valid = int(np.argmax(valid_mask_flat))
    filled[filled < 0] = first_valid
    return filled


def _flat_env_major(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    """Flatten a (T, N, ...) slice [start:end) into env-major order 1D array.

    Returns an array shaped (N*T,) for 2D inputs. Higher dims are flattened
    by the caller as needed after converting to torch.
    """
    return arr[start:end].transpose(1, 0).reshape(-1)


def _normalize_advantages(advantages: np.ndarray, eps: float) -> np.ndarray:
    """Return advantage array normalized across all elements with epsilon stability."""
    adv_flat = advantages.reshape(-1)
    return (advantages - adv_flat.mean()) / (adv_flat.std() + float(eps))


# -----------------------------
# Shared return/advantage utils
# -----------------------------
def compute_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute simple discounted Monte Carlo returns for a single trajectory.

    Parameters
    - rewards: shape (T,), rewards collected along the trajectory
    - gamma: discount factor

    Returns
    - returns: shape (T,), R_t = r_t + gamma * R_{t+1}

    Implementation detail: delegates to the batched variant for consistency.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    # Shape to (T, 1) and use no terminals/timeouts so accumulator never resets
    rewards_b = rewards.reshape(-1, 1)
    T = rewards_b.shape[0]
    dones_b = np.zeros((T, 1), dtype=bool)
    timeouts_b = np.zeros((T, 1), dtype=bool)
    returns_b = compute_batched_mc_returns(rewards_b, dones_b, timeouts_b, gamma)
    return returns_b.reshape(-1)


def compute_gae_advantages_and_returns(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: Optional[np.ndarray],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE(λ) advantages and returns for a single-trajectory (T,) case.

    Delegates to the batched implementation with batch size 1 to ensure
    consistent behavior with the collector.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    dones = np.asarray(dones, dtype=bool).reshape(-1, 1)
    if timeouts is None:
        timeouts = np.zeros_like(dones, dtype=bool)
    else:
        timeouts = np.asarray(timeouts, dtype=bool).reshape(-1, 1)

    last_values = np.asarray([float(last_value)], dtype=np.float32)

    adv_b, ret_b = compute_batched_gae_advantages_and_returns(
        values=values,
        rewards=rewards,
        dones=dones,
        timeouts=timeouts,
        last_values=last_values,
        bootstrapped_next_values=None,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return adv_b.reshape(-1), ret_b.reshape(-1)


def compute_batched_mc_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: Optional[np.ndarray],
    gamma: float,
) -> np.ndarray:
    """Compute discounted Monte Carlo-style returns for batched trajectories.

    Shapes
    - rewards: (T, N)
    - dones: (T, N) [bool]
    - timeouts: (T, N) [bool] or None

    Behavior matches the collector logic:
    - Resets the return accumulator only on real terminals (done and not timeout)
    - Timeouts are treated as non-terminals (no bootstrap is added here)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    timeouts = _canonicalize_timeouts(dones, timeouts)

    T, n_envs = rewards.shape
    returns_buf = np.zeros_like(rewards, dtype=np.float32)
    returns_acc = np.zeros(n_envs, dtype=np.float32)

    non_terminal = _non_terminal_float_mask(dones, timeouts)

    for t in range(T - 1, -1, -1):
        returns_acc = rewards[t] + gamma * returns_acc * non_terminal[t]
        returns_buf[t] = returns_acc
    return returns_buf


def compute_batched_gae_advantages_and_returns(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: Optional[np.ndarray],
    last_values: np.ndarray,
    bootstrapped_next_values: Optional[np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute batched GAE(λ) advantages and returns for (T, N) rollouts.

    - values, rewards, dones, timeouts have shape (T, N)
    - last_values has shape (N,) and bootstraps the final step per env
    - bootstrapped_next_values, if provided, has shape (T, N) and is used
      as the next value at steps that were truncated by a time limit
      (i.e., where timeouts[t, env] is True).
    - Returns are computed as advantages + values (PPO target convention).
    """
    values = np.asarray(values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    timeouts = _canonicalize_timeouts(dones, timeouts)
    last_values = np.asarray(last_values, dtype=np.float32)

    T, n_envs = rewards.shape

    # next_values: shift values by one step in time dimension; last row uses last_values
    next_values = np.zeros_like(values, dtype=np.float32)
    if T > 1:
        next_values[:-1] = values[1:]
    next_values[-1] = last_values

    # If bootstrapped_next_values provided, override next_values at timeout steps
    if bootstrapped_next_values is not None:
        bootstrapped_next_values = np.asarray(bootstrapped_next_values, dtype=np.float32)
        next_values = np.where(timeouts, bootstrapped_next_values, next_values)

    # Real terminals are env terminations that are not timeouts
    non_terminal = _non_terminal_float_mask(dones, timeouts)

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(n_envs, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * non_terminal[t] - values[t]
        gae = delta + gamma * gae_lambda * gae * non_terminal[t]
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class RolloutTrajectory(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_prob: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor


class RollingWindow:
    """O(1) rolling-window aggregator with deque semantics for len() and append().

    - Maintains a fixed-size window of recent values.
    - Tracks running sum for O(1) mean() updates.
    - Exposes append(value) and __len__ for compatibility with tests that
      assert on the number of stored items.
    """

    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("RollingWindow maxlen must be > 0")
        self._maxlen = int(maxlen)
        self._dq = deque()
        self._sum = 0.0

    def append(self, value: float) -> None:
        if len(self._dq) == self._maxlen:
            oldest = self._dq.popleft()
            self._sum -= float(oldest)
        self._dq.append(value)
        self._sum += float(value)

    def mean(self) -> float:
        n = len(self._dq)
        if n == 0:
            return 0.0
        return self._sum / n

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return len(self._dq) > 0


class RunningStats:
    """Constant-time mean/std aggregates for streaming updates (scalar over values).

    Tracks count, sum, and sum of squares. Avoids per-sample Python overhead by
    accepting array inputs and updating in vectorized form.
    """

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float32).ravel()
        if v.size == 0:
            return
        self.count += int(v.size)
        self.sum += float(v.sum())
        self.sumsq += float((v * v).sum())

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def std(self) -> float:
        if self.count == 0:
            return 0.0
        m = self.mean()
        var = max(0.0, self.sumsq / self.count - m * m)
        return float(np.sqrt(var))

class RolloutBuffer:
    """
    Persistent rollout storage to avoid reallocating buffers every collect.

    - Preallocates CPU buffers of size (maxsize, n_envs, ...).
    - begin_rollout(T) ensures a contiguous slice is available; if not, wraps
      the write position to 0 so the rollout fits contiguously.
    - Collector stores per-step data using store_* methods (tensors are
      converted to NumPy on write; conversion back to torch happens on read).
    - Provides utilities to flatten a slice into env-major tensors suitable
      for training.
    """

    def __init__(self, n_envs: int, obs_shape: Tuple[int, ...], obs_dtype: np.dtype, device: torch.device, maxsize: int) -> None:
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.device = device
        self.maxsize = int(maxsize)

        # CPU buffers
        self.obs_buf = np.zeros((self.maxsize, self.n_envs, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obs_buf = np.zeros((self.maxsize, self.n_envs, *self.obs_shape), dtype=self.obs_dtype)
        self.actions_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.int64)
        self.rewards_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)
        self.values_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)
        self.dones_buf = np.zeros((self.maxsize, self.n_envs), dtype=bool)
        self.logprobs_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)
        self.timeouts_buf = np.zeros((self.maxsize, self.n_envs), dtype=bool)
        self.bootstrapped_values_buf = np.zeros((self.maxsize, self.n_envs), dtype=np.float32)

        # write position tracking
        self.pos = 0
        self.size = 0  # number of valid steps currently stored (for info)

    def begin_rollout(self, T: int) -> int:
        """Ensure there is contiguous space for T steps; reset to 0 if needed.

        Returns the starting index where the rollout should write.
        """
        if T > self.maxsize:
            raise ValueError(f"Rollout length T={T} exceeds buffer maxsize={self.maxsize}")
        if self.pos + T > self.maxsize:
            self.pos = 0
        start = self.pos
        self.pos += T
        self.size = max(self.size, self.pos)
        return start

    # -------- Per-step storage helpers --------
    def store_tensors(self, idx: int, obs_t: torch.Tensor, actions_t: torch.Tensor, logps_t: torch.Tensor, values_t: torch.Tensor) -> None:
        """Store per-step tensors by converting to NumPy and writing CPU buffers.

        This keeps collection in NumPy primitives; conversion back to torch
        happens only when flattening for returned trajectories.
        """
        # Observations
        obs_np = obs_t.detach().cpu().numpy()
        if obs_np.shape == (self.n_envs, *self.obs_shape):
            self.obs_buf[idx] = obs_np
        else:
            # Attempt to reshape if model returned flat observations
            self.obs_buf[idx] = obs_np.reshape(self.n_envs, *self.obs_shape)

        # Actions, logprobs, values
        self.actions_buf[idx] = actions_t.detach().cpu().numpy().astype(np.int64)
        self.logprobs_buf[idx] = logps_t.detach().cpu().numpy().astype(np.float32)
        self.values_buf[idx] = values_t.detach().cpu().numpy().astype(np.float32)

    def store_cpu_step(
        self,
        idx: int,
        obs_np: np.ndarray,
        next_obs_np: np.ndarray,
        actions_np: np.ndarray,
        rewards_np: np.ndarray,
        dones_np: np.ndarray,
        timeouts_np: np.ndarray,
    ) -> None:
        # Keep shapes consistent with buffers; do not force extra dims for scalar observations
        if self.obs_shape == ():
            obs_np = np.asarray(obs_np).reshape(self.n_envs)
            next_obs_np = np.asarray(next_obs_np).reshape(self.n_envs)
        self.obs_buf[idx] = obs_np
        self.next_obs_buf[idx] = next_obs_np
        self.actions_buf[idx] = actions_np
        self.rewards_buf[idx] = rewards_np
        self.dones_buf[idx] = dones_np
        self.timeouts_buf[idx] = timeouts_np

    def copy_tensors_to_cpu(self, start: int, end: int) -> None:
        """Compatibility no-op (historical: tensors already live on CPU)."""
        return

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
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        states = obs_t.transpose(0, 1).reshape(n_envs * T, -1)

        def _flat_env_major_cpu_to_torch(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(_flat_env_major(arr, start, end), dtype=dtype, device=self.device)

        actions = _flat_env_major_cpu_to_torch(self.actions_buf, torch.int64)
        logps = _flat_env_major_cpu_to_torch(self.logprobs_buf, torch.float32)
        values = _flat_env_major_cpu_to_torch(self.values_buf, torch.float32)

        rewards = torch.as_tensor(_flat_env_major(self.rewards_buf, start, end), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(_flat_env_major(self.dones_buf, start, end), dtype=torch.bool, device=self.device)
        advantages = torch.as_tensor(_flat_env_major(advantages_buf, start, end), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(_flat_env_major(returns_buf, start, end), dtype=torch.float32, device=self.device)

        # Next observations: same env-major flattening as observations
        next_obs_tensor = torch.as_tensor(self.next_obs_buf[start:end], dtype=torch.float32, device=self.device)
        next_states = next_obs_tensor.transpose(0, 1).reshape(n_envs * T, -1)

        return RolloutTrajectory(
            observations=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_prob=logps,
            values=values,
            advantages=advantages,
            returns=returns,
            next_observations=next_states,
        )


class RolloutCollector():
    def __init__(
        self, 
        env, # Environment to collect rollouts from
        policy_model, # Policy model to collect rollouts from
        n_steps, # Number of steps to collect per rollout
        stats_window_size=100, # Size of the rolling window for stats
        gamma: float = 0.99, # Discount factor for future rewards
        gae_lambda: float = 0.95, # GAE lambda parameter (advantage estimation smoothing)
        normalize_advantages: bool = False, # Whether to normalize advantages
        advantages_norm_eps: float = 1e-8, # Epsilon for advantages normalization
        use_gae: bool = True, # Whether to use GAE (Generalized Advantage Estimation)
        buffer_maxsize: Optional[int] = None, # Maximum size of the rollout buffer
        **kwargs
):
        self.env = env
        self.policy_model = policy_model
        self.n_steps = n_steps
        self.stats_window_size = stats_window_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.advantages_norm_eps = advantages_norm_eps
        self.use_gae = use_gae
        self.normalize_advantages = normalize_advantages
        self.kwargs = kwargs
        self.buffer_maxsize = buffer_maxsize

        # State tracking
        self.device = _device_of(policy_model)
        self.n_envs = env.num_envs

        # Running average stats (windowed) with O(1) mean updates
        self.rollout_fpss = RollingWindow(stats_window_size)
        self.episode_reward_deque = RollingWindow(stats_window_size)
        self.episode_length_deque = RollingWindow(stats_window_size)
        self.env_episode_reward_deques = [RollingWindow(stats_window_size) for _ in range(self.n_envs)]
        self.env_episode_length_deques = [RollingWindow(stats_window_size) for _ in range(self.n_envs)]

        # Immediate episode stats (most recent completed episode in any env)
        self._last_episode_reward = 0.0
        self._last_episode_length = 0

        # Lightweight running statistics (avoid per-sample Python overhead)
        # Obs/reward running stats over all seen samples (scalar over all dims)
        self._obs_stats = RunningStats()
        self._rew_stats = RunningStats()

        # Action histogram (discrete); grows dynamically as needed
        self._action_counts = None

        # Baseline stats for REINFORCE (global running mean/std over returns)
        self._base_stats = RunningStats()

        self.total_rollouts = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.rollout_steps = 0
        self.rollout_episodes = 0

        # Current observations - initialize on first collect
        self.obs = None

        # Persistent rollout buffer (lazy init when first obs is known)
        self._buffer = None

        # Reusable arrays to avoid per-step allocations
        self._step_timeouts = np.zeros(self.n_envs, dtype=bool)
        # Optional index remapping for MC masking to keep dataset length stable
        self._last_rollout_index_map = None

    # -------- Small private helpers --------
    def _predict_values_np(self, obs_batch: np.ndarray) -> np.ndarray:
        """Run critic on a numpy batch and return float32 numpy array (squeezed)."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        vals = self.policy_model.predict_values(obs_t).detach().cpu().numpy().astype(np.float32)
        return vals.squeeze()

    def _bootstrap_timeouts_batch(self, start: int, terminal_obs_info: list[tuple[int, int, np.ndarray]]):
        """Predict values for all truncated terminal observations and write to buffer.

        terminal_obs_info contains tuples of (step_idx, env_idx, terminal_obs).
        """
        if not terminal_obs_info:
            return
        term_obs_batch = np.stack([info[2] for info in terminal_obs_info])
        batch_values = self._predict_values_np(term_obs_batch)
        batch_values = np.atleast_1d(batch_values)
        for i, (step_idx_term, env_idx, _) in enumerate(terminal_obs_info):
            self._buffer.bootstrapped_values_buf[start + step_idx_term, env_idx] = float(batch_values[i])

    def _update_action_histogram(self, actions_flat: np.ndarray) -> None:
        """Update cumulative discrete action counts from a flat int array."""
        if actions_flat.size == 0:
            return
        amax = int(actions_flat.max())
        if self._action_counts is None:
            self._action_counts = np.zeros(amax + 1, dtype=np.int64)
        elif amax >= self._action_counts.shape[0]:
            new_counts = np.zeros(amax + 1, dtype=np.int64)
            new_counts[: self._action_counts.shape[0]] = self._action_counts
            self._action_counts = new_counts
        binc = np.bincount(actions_flat, minlength=self._action_counts.shape[0])
        self._action_counts[: len(binc)] += binc

    def _process_done_infos(
        self,
        *,
        done_indices: np.ndarray,
        infos: list,
        step_idx: int,
        timeouts: np.ndarray,
        terminal_obs_info: list[tuple[int, int, np.ndarray]],
    ) -> None:
        """Process episode-complete infos for all done envs at a step.

        Updates deques, last-episode stats, and collects timeout terminal observations.
        """
        if len(done_indices) == 0:
            return
        for idx in done_indices:
            info = infos[idx]
            episode = info['episode']
            r = episode['r']
            l = episode['l']
            self.episode_reward_deque.append(r)
            self.episode_length_deque.append(l)
            self.env_episode_reward_deques[idx].append(r)
            self.env_episode_length_deques[idx].append(l)

            # Track immediate last episode stats (latest wins if multiple end simultaneously)
            try:
                self._last_episode_reward = float(episode.get('r', 0.0))
            except Exception:
                self._last_episode_reward = 0.0
            try:
                self._last_episode_length = int(episode.get('l', 0))
            except Exception:
                self._last_episode_length = 0

            # TimeLimit truncated bootstrap bookkeeping
            if info.get("TimeLimit.truncated"):
                timeouts[idx] = True
                terminal_obs_info.append((step_idx, idx, info["terminal_observation"]))

    @torch.inference_mode()
    def collect(self, *args, **kwargs):
        with inference_ctx(self.policy_model):
            return self._collect(*args, **kwargs)

    def _collect(self, deterministic=False):
        """Collect a single rollout and return trajectories and stats."""
        # Sync collector/buffer device with the policy model's current device.
        # Lightning may move the module after this collector is constructed.
        current_device = _device_of(self.policy_model)
        if current_device != self.device:
            self.device = current_device
            if self._buffer is not None:
                self._buffer.device = self.device

        # Initialize environment if needed
        if self.obs is None:
            self.obs = self.env.reset()

        # Lazy-init persistent buffer once we know obs shape/dtype
        if self._buffer is None:
            # For discrete observations, VecEnv returns (n_envs,)
            # Treat them as 1-feature vectors
            if self.obs.ndim == 1:
                obs_shape = (1,)
            else:
                obs_shape = self.obs.shape[1:]
            maxsize = self.buffer_maxsize if self.buffer_maxsize is not None else self.n_steps
            self._buffer = RolloutBuffer(
                n_envs=self.n_envs,
                obs_shape=obs_shape,
                obs_dtype=self.obs.dtype,
                device=self.device,
                maxsize=maxsize,
            )

        # Acquire a contiguous slice in the persistent buffer for this rollout
        start = self._buffer.begin_rollout(self.n_steps)
        end = start + self.n_steps

        # Collect terminal observations for later batch processing
        terminal_obs_info = []  # List of (step_idx, env_idx, terminal_obs)

        env_step_calls = 0
        self.rollout_steps = 0
        self.rollout_episodes = 0

        # Collect one rollout
        rollout_start = time.time()
        step_idx = 0
        while env_step_calls < self.n_steps:
            # Current observations as torch tensor (model device)
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)

            # Policy step
            actions_t, logps_t, values_t = self.policy_model.act(obs_t, deterministic=deterministic)

            # Persist per-step tensors (converted to NumPy inside)
            self._buffer.store_tensors(start + step_idx, obs_t, actions_t, logps_t, values_t)

            # Only transfer actions to CPU for environment step
            actions_np = actions_t.detach().cpu().numpy()

            # Environment step
            next_obs, rewards, dones, infos = self.env.step(actions_np)

            # Fast episode info processing - reuse timeout array
            timeouts = self._step_timeouts
            timeouts.fill(False)

            # Find all done environments at once
            done_indices = np.where(dones)[0]

            if len(done_indices) > 0:
                self._process_done_infos(
                    done_indices=done_indices,
                    infos=infos,
                    step_idx=step_idx,
                    timeouts=timeouts,
                    terminal_obs_info=terminal_obs_info,
                )

            # Persist environment outputs for this step
            self._buffer.store_cpu_step(
                start + step_idx,
                self.obs,
                next_obs,
                actions_np,
                rewards,
                dones,
                timeouts,
            )

            # Advance
            self.obs = next_obs
            env_step_calls += 1
            self.rollout_steps += self.n_envs
            self.rollout_episodes += int(dones.sum())
            step_idx += 1

        last_obs = next_obs
        self.total_steps += self.rollout_steps
        self.total_episodes += self.rollout_episodes

        # Use buffers directly from persistent storage for this slice
        T = step_idx

        # Collect observation, reward and action statistics with low overhead
        # Observations: aggregate scalar stats across all dims
        obs_block = self._buffer.obs_buf[start:end]
        obs_vals = obs_block.astype(np.float32).ravel()
        self._obs_stats.update(obs_vals)

        # Rewards
        rewards_flat = self._buffer.rewards_buf[start:end].ravel()
        self._rew_stats.update(rewards_flat)

        # Actions: maintain a histogram of encountered actions
        actions_flat = self._buffer.actions_buf[start:end].ravel()
        self._update_action_histogram(actions_flat)

        # Single batch transfer of GPU tensors to CPU after rollout collection for this slice
        self._buffer.copy_tensors_to_cpu(start, end)

        # Batch process all terminal observations at once (major performance improvement)
        if terminal_obs_info:
            self._bootstrap_timeouts_batch(start, terminal_obs_info)

        # Build next_values by shifting critic values and estimating the last
        values_slice = self._buffer.values_buf[start:end]
        rewards_slice = self._buffer.rewards_buf[start:end]
        dones_slice = self._buffer.dones_buf[start:end]
        timeouts_slice = self._buffer.timeouts_buf[start:end]

        # Prepare last values for each environment for the final bootstrap
        last_values_vec = self._predict_values_np(last_obs)

        if self.use_gae:
            bootstrapped_slice = self._buffer.bootstrapped_values_buf[start:end]
            advantages_buf, returns_buf = compute_batched_gae_advantages_and_returns(
                values=values_slice,
                rewards=rewards_slice,
                dones=dones_slice,
                timeouts=timeouts_slice,
                last_values=last_values_vec,
                bootstrapped_next_values=bootstrapped_slice,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
        else:
            # Monte Carlo returns for REINFORCE (no bootstrap added here)
            returns_buf = compute_batched_mc_returns(
                rewards=rewards_slice,
                dones=dones_slice,
                timeouts=timeouts_slice,
                gamma=self.gamma,
            )

            # Always calculate baseline and advantages using global running mean
            returns_flat = returns_buf.ravel()
            self._base_stats.update(returns_flat)

            if self._base_stats.count > 0:
                baseline = self._base_stats.mean()
                advantages_buf = returns_buf - baseline
            else:
                advantages_buf = returns_buf.copy()

            # For pure Monte Carlo methods, train only on complete episode segments within
            # the collected slice. Drop the trailing partial episode for each env where the
            # episode has not terminated inside this rollout window. This ensures that the
            # return at each included step sums rewards up to a real terminal.
            try:
                real_terminal = _real_terminal_mask(dones_slice, timeouts_slice)
                T, n_envs = real_terminal.shape
                valid_mask_2d = np.zeros((T, n_envs), dtype=bool)
                for j in range(n_envs):
                    term_idxs = np.where(real_terminal[:, j])[0]
                    if term_idxs.size > 0:
                        last_term = int(term_idxs[-1])
                        # Keep all steps up to and including the last terminal
                        valid_mask_2d[: last_term + 1, j] = True
                # Flatten mask in env-major order to align with flatten_slice_env_major
                valid_mask_flat = valid_mask_2d.transpose(1, 0).reshape(-1)
                self._last_rollout_index_map = _build_idx_map_from_valid_mask(valid_mask_flat)
            except Exception:
                # If anything goes wrong, fall back to using all steps (no remap)
                self._last_rollout_index_map = None

        if self.normalize_advantages:
            advantages_buf = _normalize_advantages(advantages_buf, self.advantages_norm_eps)

        # Create final training tensors from persistent buffers
        trajectories = self._buffer.flatten_slice_env_major(start, end, advantages_buf, returns_buf)

        # For MC path, we left dataset length unchanged and stored an index map
        # to remap invalid indices at collate time.

        self.total_rollouts += 1
        rollout_elapsed = time.time() - rollout_start
        fps = len(trajectories.observations) / rollout_elapsed  # steps per second
        self.rollout_fpss.append(fps)

        return trajectories

    def slice_trajectories(self, trajectories, idxs):
        # If an index map is available for MC masking, remap indices to valid
        # positions to keep batch sizes and sampler assumptions intact.
        try:
            if self._last_rollout_index_map is not None and not self.use_gae:
                # idxs may be a list or tensor; convert to numpy indices
                import numpy as _np
                idxs_np = _np.asarray(idxs, dtype=_np.int64)
                # Safe remap into valid positions
                idxs_np = self._last_rollout_index_map[idxs_np]
                idxs = idxs_np
        except Exception:
            pass
        return RolloutTrajectory(
            observations=trajectories.observations[idxs],
            actions=trajectories.actions[idxs],
            rewards=trajectories.rewards[idxs],
            dones=trajectories.dones[idxs],
            log_prob=trajectories.log_prob[idxs],
            values=trajectories.values[idxs],
            advantages=trajectories.advantages[idxs],
            returns=trajectories.returns[idxs],
            next_observations=trajectories.next_observations[idxs]
        )

    def get_metrics(self):
        ep_rew_mean = float(self.episode_reward_deque.mean()) if self.episode_reward_deque else 0.0
        ep_len_mean = int(self.episode_length_deque.mean()) if self.episode_length_deque else 0
        rollout_fps = float(self.rollout_fpss.mean()) if self.rollout_fpss else 0.0

        # Observation statistics from running aggregates
        obs_mean = self._obs_stats.mean()
        obs_std = self._obs_stats.std()

        # Reward statistics
        reward_mean = self._rew_stats.mean()
        reward_std = self._rew_stats.std()

        # Action statistics: derive mean/std from histogram if available
        if self._action_counts is not None and self._action_counts.sum() > 0:
            idxs = np.arange(self._action_counts.shape[0], dtype=np.float32)
            total = float(self._action_counts.sum())
            mean_a = float((idxs * self._action_counts).sum() / total)
            var_a = float(((idxs - mean_a) ** 2 * self._action_counts).sum() / total)
            action_mean = mean_a
            action_std = float(np.sqrt(max(0.0, var_a)))
            action_dist = None  # keep optional distribution disabled to avoid large logs
        else:
            action_mean, action_std, action_dist = 0.0, 0.0, None

        # Baseline statistics from global aggregates
        baseline_mean = self._base_stats.mean()
        baseline_std = self._base_stats.std()

        return {
            "total_timesteps": self.total_steps,  # TODO: steps vs timesteps
            "total_episodes": self.total_episodes,
            "total_rollouts": self.total_rollouts,
            "rollout_timesteps": self.rollout_steps,
            "rollout_episodes": self.rollout_episodes,  # Renamed to avoid conflict with video logging
            "rollout_fps": rollout_fps,  # TODO: this is a mean, it shouln't be
            # Immediate last episode stats
            "ep_rew_last": float(self._last_episode_reward),
            "ep_len_last": int(self._last_episode_length),
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "action_dist": action_dist,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std
        }

    def get_action_histogram_counts(self, reset: bool = False):
        """Return a copy of the cumulative discrete action counts histogram.

        Args:
            reset: If True, zero the internal counters after returning the copy.

        Returns:
            np.ndarray | None: 1D array of counts per discrete action index, or None if no actions seen yet.
        """
        counts = self._action_counts
        if counts is None: return None
        out = counts.copy()
        if reset: self._action_counts = np.zeros_like(counts)
        return out
