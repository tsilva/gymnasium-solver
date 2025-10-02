import time
from collections import deque
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from utils.policy_ops import policy_act, policy_predict_values
from utils.torch import _device_of, inference_ctx


def _to_np(t: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return t.detach().cpu().numpy().astype(dtype, copy=False)


def _real_terminal_mask(dones: np.ndarray, timeouts: np.ndarray) -> np.ndarray:
    """Real terminal if done and not a time-limit truncation."""
    dones_b = np.asarray(dones, dtype=bool)
    timeouts_b = np.asarray(timeouts, dtype=bool)
    return np.logical_and(dones_b, ~timeouts_b)


def _non_terminal_float_mask(dones: np.ndarray, timeouts: Optional[np.ndarray]) -> np.ndarray:
    """Float32 mask of non-terminals given dones/timeouts."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    return (~real_terminal).astype(np.float32)


def _build_idx_map_from_valid_mask(valid_mask_flat: np.ndarray) -> Optional[np.ndarray]:
    """Map each position to the nearest previous valid index; None if no valid entries."""
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


def _build_valid_mask_and_index_map(
    dones: np.ndarray,
    timeouts: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build env-major valid mask up to last real terminal and its index map."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    if real_terminal.size == 0:
        return None, None
    T, n_envs = real_terminal.shape
    valid_mask_2d = np.zeros((T, n_envs), dtype=bool)
    for j in range(n_envs):
        term_idxs = np.where(real_terminal[:, j])[0]
        if term_idxs.size > 0:
            last_term = int(term_idxs[-1])
            valid_mask_2d[: last_term + 1, j] = True
    valid_mask_flat = valid_mask_2d.transpose(1, 0).reshape(-1)
    if not valid_mask_flat.any():
        return None, None
    idx_map = _build_idx_map_from_valid_mask(valid_mask_flat)
    return valid_mask_flat, idx_map


def _flat_env_major(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    """Flatten a (T,N,...) slice [start:end) into a (N*T,) env-major 1D array."""
    return arr[start:end].transpose(1, 0).reshape(-1)

def _normalize_returns(returns: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return return array normalized across all elements with epsilon stability."""
    ret_flat = returns.reshape(-1)
    return (returns - ret_flat.mean()) / (ret_flat.std() + float(eps))


def _normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return advantage array normalized across all elements with epsilon stability."""
    adv_flat = advantages.reshape(-1)
    return (advantages - adv_flat.mean()) / (adv_flat.std() + float(eps))


# -----------------------------
# Shared return/advantage utils (batched only)
# -----------------------------


def compute_batched_mc_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute discounted Monte Carlo-style returns for batched trajectories."""

    T, n_envs = rewards.shape
    returns_buf = np.zeros_like(rewards, dtype=np.float32)
    returns_acc = np.zeros(n_envs, dtype=np.float32)
    
    # Create a mask of non-terminal observations
    # (all observations that are not done and not timeout)
    non_terminal = _non_terminal_float_mask(dones, timeouts)

    for t in range(T - 1, -1, -1):
        _rewards_t = rewards[t]
        _non_terminal_t = non_terminal[t]
        _future_returns = returns_acc * _non_terminal_t
        _discounted_future_returns = gamma * _future_returns
        returns_acc = _rewards_t + _discounted_future_returns
        returns_buf[t] = returns_acc

    return returns_buf

def convert_returns_to_full_episode(
    returns: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray,
) -> np.ndarray:
    """Convert reward-to-go returns to per-episode constants in-place."""
    real_terminal = _real_terminal_mask(dones, timeouts)
    T, n_envs = returns.shape if returns.size > 0 else (0, 0)
    for j in range(n_envs):
        seg_start = 0
        seg_value = returns[seg_start, j] if T > 0 else 0.0
        for t in range(T):
            # Set the return at step t to the segment's initial return
            returns[t, j] = seg_value
            if not real_terminal[t, j]:
                continue
            seg_start = t + 1
            if seg_start >= T:
                break
            seg_value = returns[seg_start, j]
    return returns

def compute_batched_gae_advantages_and_returns(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray,
    last_values: np.ndarray,
    bootstrapped_next_values: Optional[np.ndarray],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute batched GAE(λ) advantages and returns for (T, N) rollouts."""
    values = np.asarray(values, dtype=np.float32) # TODO: why this?
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    timeouts = np.asarray(timeouts, dtype=bool)
    last_values = np.asarray(last_values, dtype=np.float32)

    T, n_envs = rewards.shape

    # next_values: shift values by one step in time dimension; last row uses last_values
    next_values = np.zeros_like(values, dtype=np.float32)
    if T > 1: next_values[:-1] = values[1:]
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
    logprobs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_observations: torch.Tensor


class RollingWindow:
    """O(1) rolling window with deque semantics and constant-time mean()."""

    def __init__(self, maxlen: int):
        if maxlen <= 0: raise ValueError("RollingWindow maxlen must be > 0")
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
        if n == 0: return 0.0
        return self._sum / n

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return len(self._dq) > 0


class RunningStats:
    """Constant-time mean/std aggregates for streaming numeric updates."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_squared = 0.0

    def update(self, values: np.ndarray) -> None:
        # Fast path: skip stats collection if values are empty
        if values.size == 0: return

        self.count += values.size

        # Work with flattened view directly (no copy)
        vals_flat = values.ravel()

        # Convert to float32 only if needed
        if vals_flat.dtype != np.float32:
            vals_flat = vals_flat.astype(np.float32)

        # Accumulate stats (keep in float32 for speed)
        self.sum += float(vals_flat.sum())
        self.sum_squared += float((vals_flat * vals_flat).sum())

    def mean(self) -> float:
        if self.count == 0: return 0.0
        return self.sum / self.count

    def std(self) -> float:
        if self.count == 0: return 0.0
        mean = self.mean()
        var = max(0.0, self.sum_squared / self.count - mean * mean)
        return float(np.sqrt(var))

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


class RolloutCollector():
    def __init__(
        self, 
        env, # Environment to collect rollouts from
        policy_model, # Policy model to collect rollouts from
        n_steps, # Number of steps to collect per rollout
        *,
        stats_window_size=100, # Size of the rolling window for stats
        gamma: float = 0.99, # Discount factor for future rewards
        gae_lambda: float = 0.95, # GAE lambda parameter (advantage estimation smoothing)
        returns_type: Optional[str] = None, # Which returns type to use (compat: accepts aliases)
        normalize_returns: bool = False, # Whether to normalize returns
        advantages_type: Optional[str] = None, # Which advantages type to use (compat: accepts aliases)
        normalize_advantages: bool = False, # Whether to normalize advantages
        buffer_maxsize: Optional[int] = None, # Maximum size of the rollout buffer
        mc_treat_timeouts_as_terminals: bool = True,
        use_gae: Optional[bool] = None,
        **kwargs
    ) -> None:
        self.env = env
        self.policy_model = policy_model
        self.n_steps = n_steps
        self.stats_window_size = stats_window_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # Normalize type specifiers and support legacy flags
        def _to_str(x):
            try:
                return str(x.value)  # Enum-like
            except AttributeError:
                return None if x is None else str(x)

        rtype = _to_str(returns_type)
        atype = _to_str(advantages_type)
        # If unspecified, infer from use_gae flag
        if rtype is None or atype is None:
            if bool(use_gae):
                rtype = rtype or "gae:rtg"
                atype = atype or "gae"
            else:
                rtype = rtype or "mc:rtg"
                atype = atype or "baseline"
        # Accept loose aliases (including Enum member names)
        alias_map = {
            # Human-friendly shorthands
            "episode": "mc:episode",
            "reward_to_go": "mc:rtg",
            "rtg": "mc:rtg",
            # Enum member names (from utils.config.Config enums)
            "mc_episode": "mc:episode",
            "mc_rtg": "mc:rtg",
            "gae_rtg": "gae:rtg",
            # Advantages
            "gae": "gae",
            "baseline": "baseline",
        }
        rtype = alias_map.get(rtype, rtype)
        atype = alias_map.get(atype, atype)

        self.returns_type = rtype
        self.normalize_returns = normalize_returns
        self.advantages_type = atype
        self.normalize_advantages = normalize_advantages
        self.mc_treat_timeouts_as_terminals = mc_treat_timeouts_as_terminals # TODO: review this
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
        self._best_episode_reward = -float('inf')

        # Lightweight running statistics (avoid per-sample Python overhead)
        # Obs/reward running stats over all seen samples (scalar over all dims)
        self._obs_stats = RunningStats()
        self._rew_stats = RunningStats()
        self._base_stats = RunningStats()
        self._adv_stats = RunningStats()
        self._ret_stats = RunningStats()

        # Action histogram (discrete); grows dynamically as needed
        self._action_counts = None

        self.total_rollouts = 0
        self.total_steps = 0
        # Vectorized step counters: one increment per env.step() call across the vector
        self.total_vec_steps = 0
        self.total_episodes = 0
        self.rollout_steps = 0
        self.rollout_vec_steps = 0
        self.rollout_episodes = 0

        # Current observations - initialize on first collect
        self.obs = None

        # Persistent rollout buffer (lazy init when first obs is known)
        self._buffer = None

        # Reusable arrays to avoid per-step allocations
        self._step_timeouts = np.zeros(self.n_envs, dtype=bool)

        # Optional index remapping for MC masking to keep dataset length stable
        # TODO: why this?
        self._last_rollout_index_map = None

        # Collect terminal observations for later batch processing
        self.terminal_obs_info = []  # List of (step_idx, env_idx, terminal_obs)

        # Episodes completed since last pop (env_idx, reward, length, was_timeout)
        # Used by evaluate_episodes() to compute unbiased metrics from finished episodes
        self._recent_episodes = []

    # -------- Small private helpers --------
    def _predict_values_np(self, obs_batch: np.ndarray) -> np.ndarray:
        """Run critic on a numpy batch and return float32 numpy array (squeezed)."""
        obs_t = torch.as_tensor(obs_batch, device=self.device)
        vals = policy_predict_values(self.policy_model, obs_t).detach().cpu().numpy().astype(np.float32)
        return vals.squeeze()

    def _bootstrap_timeouts_batch(self, start: int):
        """Predict values for truncated terminals and write to buffer."""
        if not self.terminal_obs_info: return
        # Coerce terminal observations to match buffer obs_shape (e.g., HWC -> CHW)
        exp = tuple(self._buffer.obs_shape)
        coerced_list = []
        for (_, _, obs) in self.terminal_obs_info:
            arr = np.asarray(obs)
            if len(exp) == 3:
                C, H, W = int(exp[0]), int(exp[1]), int(exp[2])
                if arr.ndim == 3:
                    # HWC -> CHW
                    if arr.shape == (H, W, C):
                        arr = np.transpose(arr, (2, 0, 1))
                    # Already CHW
                    elif arr.shape == (C, H, W):
                        pass
                    # Grayscale H,W -> add channel
                    elif arr.shape == (H, W) and C == 1:
                        arr = arr[None, ...]
                elif arr.ndim == 2 and C == 1 and arr.shape == (H, W):
                    arr = arr[None, ...]
            # For non-image shapes, keep as-is
            coerced_list.append(arr)
        term_obs_batch = np.stack(coerced_list)
        batch_values = self._predict_values_np(term_obs_batch)
        batch_values = np.atleast_1d(batch_values)
        for i, (step_idx_term, env_idx, _) in enumerate(self.terminal_obs_info):
            self._buffer.bootstrapped_values_buf[start + step_idx_term, env_idx] = float(batch_values[i])

    def _update_action_histogram(self, actions_flat: np.ndarray) -> None:
        """Update cumulative discrete action counts from a flat int array."""
        if actions_flat.size == 0: return
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
        done_idxs: np.ndarray,
        infos: dict,
        step_idx: int
    ) -> None:
        """Process episode-complete infos for all done envs at a step."""
        assert len(done_idxs) > 0, "No done environments at a step"

        # Gymnasium VectorEnv with RecordEpisodeStatistics stores episode stats as:
        # info['episode'] = {'r': array([reward1, reward2, ...]), 'l': array([len1, len2, ...]), ...}
        # info['_episode'] = array([True, False, ...])  # mask of which envs finished episodes
        episode_dict = infos.get("episode", None)
        episode_mask = infos.get("_episode", None)

        # Also check for final_info/final_observation (alternative Gymnasium format)
        final_infos = infos.get("final_info", None)
        final_observations = infos.get("final_observation", None)

        # Process episode-complete infos for all done envs at a step
        for idx, env_idx in enumerate(done_idxs):
            # Try to get episode stats from the Gymnasium RecordEpisodeStatistics format
            episode_reward = 0.0
            episode_length = 0

            if episode_dict is not None and episode_mask is not None:
                # RecordEpisodeStatistics format: arrays indexed by env_idx
                if episode_mask[env_idx]:
                    episode_reward = float(episode_dict['r'][env_idx])
                    episode_length = int(episode_dict['l'][env_idx])
            elif final_infos is not None:
                # Alternative format: list of info dicts
                info = final_infos[idx] if idx < len(final_infos) else {}
                if info is not None:
                    episode = info.get('episode', {})
                    episode_reward = episode.get('r', 0.0)
                    episode_length = episode.get('l', 0)

            # Add episode data to deques (for stats tracking)
            self.episode_reward_deque.append(episode_reward)
            self.episode_length_deque.append(episode_length)
            self.env_episode_reward_deques[env_idx].append(episode_reward)
            self.env_episode_length_deques[env_idx].append(episode_length)

            # Track last episode stats
            self._last_episode_reward = episode_reward
            self._last_episode_length = episode_length

            # Track best episode reward
            self._best_episode_reward = max(self._best_episode_reward, episode_reward)

            # In case the episode ended due to time limit (truncation) then
            # retrieve the last observation (truncated episode observations are the
            # observation from the next episode, we'll need the actual last
            # observation to bootstrap the value function)
            # Note: self._step_timeouts is set before calling this method based on truncated array
            is_timeout = bool(self._step_timeouts[env_idx])
            if is_timeout:
                # Try to get terminal observation from various formats
                terminal_observation = None

                # Check final_observations list format
                if final_observations is not None and idx < len(final_observations):
                    terminal_observation = final_observations[idx]

                # Check if there's a final_observation dict indexed by env_idx
                if terminal_observation is None and "final_observation" in infos:
                    final_obs = infos["final_observation"]
                    if isinstance(final_obs, np.ndarray) and final_obs.ndim > 1:
                        # Array format: shape (n_envs, *obs_shape)
                        terminal_observation = final_obs[env_idx]

                if terminal_observation is not None:
                    self.terminal_obs_info.append((step_idx, env_idx, terminal_observation))

            # Record this finished episode for evaluation accounting
            self._recent_episodes.append(
                (
                    int(env_idx),
                    float(episode_reward),
                    int(episode_length),
                    is_timeout,
                )
            )

    @torch.inference_mode()
    def collect(self, *args, **kwargs):
        """Collect one rollout slice using the current policy.

        Wraps the internal implementation with an inference context to disable
        autograd and keep parity with Lightning’s device moves.
        """
        with inference_ctx(self.policy_model):
            return self._collect(*args, **kwargs)

    # -------- Internal phases (clarity; no per-step calls) --------
    def _sync_device_and_prepare_buffers(self) -> None:
        """Sync device with model, ensure initial obs and persistent buffer exist."""

        if self.obs is not None: return

        current_device = _device_of(self.policy_model)
        self.device = current_device

        # If this is the first collection, reset the 
        # environment to get the first observation
        self.obs, _ = self.env.reset()

        # For discrete observations, VecEnv returns (n_envs,), treat as 1-feature
        obs_shape = (1,) if self.obs.ndim == 1 else self.obs.shape[1:]
        
        # TODO: review this
        maxsize = self.buffer_maxsize if self.buffer_maxsize is not None else self.n_steps

        self._buffer = RolloutBuffer(
            n_envs=self.n_envs,
            obs_shape=obs_shape,
            obs_dtype=self.obs.dtype,
            device=self.device,
            maxsize=maxsize,
        )

    def _update_running_stats_after_rollout(self, start: int, end: int) -> None:
        """Update obs/reward/action stats and perform any deferred CPU copies."""
        # Pass slices directly - RunningStats.update now handles dtypes efficiently
        self._obs_stats.update(self._buffer.obs_buf[start:end])
        self._rew_stats.update(self._buffer.rewards_buf[start:end])

        # Add actions to action histogram
        actions_block = self._buffer.actions_buf[start:end]
        if actions_block.dtype != np.int64:
            actions_block = actions_block.astype(np.int64)
        self._update_action_histogram(actions_block.ravel())

    def _compute_targets(self, start: int, end: int, last_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns for the slice [start, end)."""
        values_slice = self._buffer.values_buf[start:end]
        rewards_slice = self._buffer.rewards_buf[start:end]
        dones_slice = self._buffer.dones_buf[start:end]
        timeouts_slice = self._buffer.timeouts_buf[start:end]

        # Track valid mask for MC-style targets so advantage stats ignore trailing partials
        valid_mask_flat_for_stats: Optional[np.ndarray] = None

        if self.returns_type == "gae:rtg" and self.advantages_type == "gae":
            last_values_vec = self._predict_values_np(last_obs)
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
        elif self.returns_type in ["mc:episode", "mc:rtg"]:
            # Monte Carlo returns for REINFORCE (no bootstrap added here)
            # Optionally treat time-limit truncations as terminals when not bootstrapping
            # to avoid return leakage across episode boundaries.
            # A real terminal observation is one that is "done" but not a "timeout". However
            # for methods that don't bootstrap, we treat timeouts as terminals to avoid
            # return leakage across episode boundaries.
            _timeouts_slice = timeouts_slice
            if self.mc_treat_timeouts_as_terminals: _timeouts_slice = np.zeros_like(timeouts_slice, dtype=bool)
            returns_buf = compute_batched_mc_returns(
                rewards=rewards_slice,
                dones=dones_slice,
                timeouts=_timeouts_slice,
                gamma=self.gamma,
            )

            # If requested, convert reward-to-go returns into full-episode returns
            # by making all timesteps within the same episode segment share the
            # segment's initial return (constant across the segment).
            if self.returns_type == "mc:episode":
                returns_buf = convert_returns_to_full_episode(
                    returns=returns_buf,
                    dones=dones_slice,
                    timeouts=_timeouts_slice,
                )

            # Build a valid-mask index map to exclude trailing partial episodes
            valid_mask_flat, idx_map = _build_valid_mask_and_index_map(dones_slice, _timeouts_slice)
            self._last_rollout_index_map = idx_map

            # Update global baseline using only valid (non-trailing) returns
            if valid_mask_flat is not None:
                returns_flat_env_major = returns_buf.transpose(1, 0).reshape(-1)
                self._base_stats.update(returns_flat_env_major[valid_mask_flat])
                # Record mask to compute advantage stats later on valid positions only
                valid_mask_flat_for_stats = valid_mask_flat

            advantages_buf = returns_buf
            if self.advantages_type == "baseline":
                baseline = self._base_stats.mean()
                advantages_buf = returns_buf - baseline
        else:
            raise ValueError(f"Invalid returns_type: {self.returns_type} and advantages_type: {self.advantages_type}")

        # Normalize returns if requested
        if self.normalize_returns:
            returns_buf = _normalize_returns(returns_buf) # TODO: take into account unfinished episodes?

        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages_buf = _normalize_advantages(advantages_buf)

        # Update running return/advantage stats (post-normalization so it reflects training distribution)
        ret_flat_env_major = returns_buf.transpose(1, 0).reshape(-1)
        if valid_mask_flat_for_stats is not None:
            self._ret_stats.update(ret_flat_env_major[valid_mask_flat_for_stats])
        else:
            self._ret_stats.update(ret_flat_env_major)

        # Advantages
        adv_flat_env_major = advantages_buf.transpose(1, 0).reshape(-1)
        if valid_mask_flat_for_stats is not None:
            self._adv_stats.update(adv_flat_env_major[valid_mask_flat_for_stats])
        else:
            self._adv_stats.update(adv_flat_env_major)

        return advantages_buf, returns_buf

    def _collect(self, deterministic=False):
        """Collect a single rollout and return trajectories and stats."""
        # Sync device, ensure obs and buffers are ready
        self._sync_device_and_prepare_buffers()

        # Acquire a contiguous slice in the persistent buffer for this rollout
        start = self._buffer.begin_rollout(self.n_steps)
        end = start + self.n_steps

        self.rollout_steps = 0
        self.rollout_vec_steps = 0
        self.rollout_episodes = 0

        # Collect one rollout
        rollout_start = time.time()
        for step_idx in range(self.n_steps):
            # Convert current observations to torch tensor (ship to device)
            obs_t = torch.as_tensor(self.obs, device=self.device)

            # Perform policy step to determine actions, log probabilities, and value estimates
            actions_t, logps_t, values_t = policy_act(self.policy_model, obs_t, deterministic=deterministic)

            # Extract action probabilities for visualization (if wrapper supports it)
            try:
                dist, _ = self.policy_model(obs_t)
                if hasattr(dist, 'probs'):
                    action_probs_np = dist.probs.detach().cpu().numpy()
                    if hasattr(self.env, 'set_action_probs'):
                        self.env.set_action_probs(action_probs_np)
            except Exception:
                pass

            # Perform environment step
            actions_np = actions_t.detach().cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions_np)

            # Render if environment has human rendering enabled
            if hasattr(self.env, 'render_mode') and self.env.render_mode == 'human':
                self.env.render()

            # Combine terminated and truncated into dones for backward compatibility
            dones = np.logical_or(terminated, truncated)

            # In case there are any done environments, process the episode info
            self._step_timeouts.fill(False)
            self._step_timeouts[truncated] = True

            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                self._process_done_infos(done_idxs=done_indices, infos=infos, step_idx=step_idx)

            # Persist environment outputs for this step
            logps_np = _to_np(logps_t, np.float32)
            values_np = _to_np(values_t, np.float32)
            self._buffer.add(
                start + step_idx,
                self.obs,
                next_obs,
                actions_np,
                logps_np,
                values_np,
                rewards,
                dones,
                self._step_timeouts,
            )

            # Next observation is now current observation
            self.obs = next_obs

            # Advance rollout stats
            self.rollout_steps += self.n_envs
            self.rollout_vec_steps += 1
            self.rollout_episodes += int(dones.sum())

        last_obs = next_obs
        self.total_steps += self.rollout_steps
        self.total_vec_steps += self.rollout_vec_steps
        self.total_episodes += self.rollout_episodes

        # Update stats based on the collected slice
        self._update_running_stats_after_rollout(start, end)

        # Batch process all terminal observations at once
        if self.terminal_obs_info: 
            self._bootstrap_timeouts_batch(start)

        # Compute advantages and returns for this slice
        advantages_buf, returns_buf = self._compute_targets(start, end, last_obs)

        # Create final training tensors from persistent buffers
        trajectories = self._buffer.flatten_slice_env_major(start, end, advantages_buf, returns_buf)

        self.total_rollouts += 1
        rollout_elapsed = time.time() - rollout_start
        fps = len(trajectories.observations) / rollout_elapsed
        self.rollout_fpss.append(fps)

        return trajectories

    @torch.inference_mode()
    def evaluate_episodes(
        self,
        *,
        n_episodes: int,
        deterministic: bool = True,
        timeout_seconds: Optional[float] = None,
    ) -> dict:
        """Evaluate policy for exactly N episodes using this collector's env.

        Uses the collector's `collect()` method to step environments and leverages
        its episode bookkeeping. Computes unbiased means from finished episodes only
        (ignores trailing partials). If `max_steps_per_episode` or `timeout_seconds`
        are provided, they are currently ignored to avoid invasive changes.
        """
        assert hasattr(self.env, "num_envs"), "Environment must be vectorized (have num_envs)"

        # Helper to distribute targets evenly across env ranks (stable-baselines style)
        def _balanced_targets(n_envs: int, total_episodes: int):
            if total_episodes <= 0:
                return [0] * n_envs
            base = int(total_episodes) // int(n_envs)
            rem = int(total_episodes) % int(n_envs)
            return [base + (1 if i < rem else 0) for i in range(int(n_envs))]

        n_envs = int(self.env.num_envs)
        per_env_targets = _balanced_targets(n_envs, int(n_episodes))
        per_env_counts = [0] * n_envs

        # Running aggregates computed from finished episodes only
        total_reward_sum = 0.0
        total_length_sum = 0
        total_timesteps = 0
        total_vec_steps = 0

        # Start fresh episodes: force a reset/alloc on next collection
        self.obs = None
        self._recent_episodes = []
        self._sync_device_and_prepare_buffers()

        start_time = time.time()
        while True:
            # Stop once all envs reached their per-env target counts
            if all(count >= tgt for count, tgt in zip(per_env_counts, per_env_targets)):
                break

            # Collect one rollout worth of steps using the current policy
            traj = self.collect(deterministic=deterministic)
            total_timesteps += int(traj.observations.shape[0])
            # Each collect call advanced this many vector steps
            total_vec_steps += int(self.rollout_vec_steps)

            # Consume finished episodes recorded during this rollout
            recent_eps = getattr(self, "_recent_episodes", None) or []
            self._recent_episodes = []
            for env_idx, ep_rew, ep_len, _was_timeout in recent_eps:
                # Skip episodes for envs that already reached their target
                if per_env_counts[env_idx] >= per_env_targets[env_idx]:
                    continue
                total_reward_sum += float(ep_rew)
                total_length_sum += int(ep_len)
                per_env_counts[env_idx] += 1

            # Optional wall-clock timeout (best-effort; breaks between rollouts)
            if timeout_seconds is not None and (time.time() - start_time) >= float(timeout_seconds):
                break

        total_episodes_collected = int(sum(per_env_counts))
        ep_rew_mean = float(total_reward_sum / total_episodes_collected) if total_episodes_collected > 0 else 0.0
        ep_len_mean = float(total_length_sum / total_episodes_collected) if total_episodes_collected > 0 else 0.0

        base_metrics = self.get_metrics()
        
        # TODO: this is a hack, make sure _dist is being excluded from logging
        base_metrics.pop("action_dist")

        return {
            **base_metrics,
            "cnt/total_episodes": total_episodes_collected,
            "cnt/total_env_steps": int(total_timesteps),
            "cnt/total_vec_steps": int(total_vec_steps),
            "roll/ep_rew/mean": ep_rew_mean,
            "roll/ep_len/mean": float(ep_len_mean),
        }

    def slice_trajectories(self, trajectories, idxs):
        """Return a view of the rollout trajectory at the given indices.

        When using Monte Carlo returns (no GAE), incomplete trailing segments
        are masked out by remapping indices to the nearest previous valid
        position so batch shapes stay stable across samplings.
        """
        if self._last_rollout_index_map is not None and not self.advantages_type == "gae":
            # idxs may be a list or tensor; convert to numpy indices
            idxs_np = np.asarray(idxs, dtype=np.int64)
            # Safe remap into valid positions
            idxs_np = self._last_rollout_index_map[idxs_np]
            idxs = idxs_np

        # TODO: should I call this Trajectory or Trajectories?
        return RolloutTrajectory(
            observations=trajectories.observations[idxs],
            actions=trajectories.actions[idxs],
            rewards=trajectories.rewards[idxs],
            dones=trajectories.dones[idxs],
            logprobs=trajectories.logprobs[idxs],
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
            # Return action counts for histogram logging (will be filtered from regular metrics)
            action_dist = self._action_counts.copy()
        else:
            action_mean, action_std, action_dist = 0.0, 0.0, None

        # Baseline statistics from global aggregates
        baseline_mean = self._base_stats.mean()
        baseline_std = self._base_stats.std()
        # Advantage statistics (post-normalization if enabled)
        adv_mean = self._adv_stats.mean()
        adv_std = self._adv_stats.std()
        # Return statistics (post-normalization if enabled)
        ret_mean = self._ret_stats.mean()
        ret_std = self._ret_stats.std()

        return {
            "cnt/total_env_steps": self.total_steps,
            "cnt/total_vec_steps": self.total_vec_steps,  # canonical step counter for history
            "cnt/total_episodes": self.total_episodes,
            "cnt/total_rollouts": self.total_rollouts,
            "roll/env_steps": self.rollout_steps,
            "roll/vec_steps": self.rollout_vec_steps,
            "roll/episodes": self.rollout_episodes,
            "roll/fps": rollout_fps,  # average fps over recent rollouts
            "roll/ep_rew/best": float(self._best_episode_reward),
            "roll/ep_rew/last": float(self._last_episode_reward),
            "roll/ep_len/last": int(self._last_episode_length),
            "roll/ep_rew/mean": ep_rew_mean,
            "roll/ep_len/mean": ep_len_mean,
            "roll/obs/mean": obs_mean,
            "roll/obs/std": obs_std,
            "roll/reward/mean": reward_mean,
            "roll/reward/std": reward_std,
            "roll/return/mean": ret_mean,
            "roll/return/std": ret_std,
            "roll/adv/mean": adv_mean,
            "roll/adv/std": adv_std,
            "roll/actions/mean": action_mean,
            "roll/actions/std": action_std,
            "action_dist": action_dist,
            "roll/baseline/mean": baseline_mean,
            "roll/baseline/std": baseline_std
        }

    def pop_recent_episodes(self) -> List[Tuple[int, float, int, bool]]:
        """Return and clear episodes finished since the last pop."""
        recent = getattr(self, "_recent_episodes", None)
        if not recent:
            return []
        episodes = list(recent)
        self._recent_episodes = []
        return episodes

    def get_action_histogram_counts(self, reset: bool = False):
        """Return a copy of discrete action counts (optionally reset internal counters)."""
        counts = self._action_counts
        if counts is None: return None
        out = counts.copy()
        if reset: self._action_counts = np.zeros_like(counts)
        return out
