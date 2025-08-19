import time
from collections import deque
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

from utils.torch_utils import _device_of, inference_ctx


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
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    T = rewards.shape[0]
    out = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def compute_gae_advantages_and_returns(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: Optional[np.ndarray],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE(λ) advantages and corresponding returns for a single env trajectory.

    Mirrors the logic used in RolloutCollector for a 1D trajectory (T,).

    Notes
    - "dones" are environment terminations (can include time-limit truncations if you OR'd them).
    - "timeouts" marks steps that were truncated by a time limit (no true terminal); if provided and the
      final step is a timeout, we bootstrap using last_value. For intermediate steps in a single-episode
      trajectory, timeouts is typically all False except possibly the last step.
    - Returns are computed as advantages + values, matching PPO-style training targets.
    """
    values = np.asarray(values, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    if timeouts is None:
        timeouts = np.zeros_like(dones, dtype=bool)
    else:
        timeouts = np.asarray(timeouts, dtype=bool)

    T = rewards.shape[0]
    # Build next_values by shifting and bootstrapping the last with last_value
    next_values = np.zeros_like(values, dtype=np.float32)
    if T > 1:
        next_values[:-1] = values[1:]
    next_values[-1] = float(last_value)

    # Real terminals are dones that are not timeouts
    real_terminal = np.logical_and(dones, ~timeouts)
    non_terminal = (~real_terminal).astype(np.float32)

    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * non_terminal[t] - values[t]
        gae = float(delta + gamma * gae_lambda * gae * non_terminal[t])
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


class RolloutTrajectory(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
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
            return torch.as_tensor(arr[start:end].transpose(1, 0).reshape(-1), dtype=dtype, device=self.device)

        actions = _flat_env_major_cpu_to_torch(self.actions_buf, torch.int64)
        logps = _flat_env_major_cpu_to_torch(self.logprobs_buf, torch.float32)
        values = _flat_env_major_cpu_to_torch(self.values_buf, torch.float32)

        def _flat_env_major_cpu_only(arr: np.ndarray) -> np.ndarray:
            return arr[start:end].transpose(1, 0).reshape(-1)
        rewards = torch.as_tensor(_flat_env_major_cpu_only(self.rewards_buf), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(_flat_env_major_cpu_only(self.dones_buf), dtype=torch.bool, device=self.device)
        advantages = torch.as_tensor(_flat_env_major_cpu_only(advantages_buf), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(_flat_env_major_cpu_only(returns_buf), dtype=torch.float32, device=self.device)

        # Next observations
        if self.next_obs_buf.ndim == 2:
            # Ensure shape matches observations: (n_envs*T, 1)
            next_states = torch.as_tensor(
                self.next_obs_buf[start:end].transpose(1, 0).reshape(-1, 1),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            next_obs_tensor = torch.as_tensor(self.next_obs_buf[start:end], dtype=torch.float32, device=self.device)
            next_states = next_obs_tensor.transpose(0, 1).reshape(n_envs * T, -1)

        return RolloutTrajectory(
            observations=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            old_log_prob=logps,
            old_values=values,
            advantages=advantages,
            returns=returns,
            next_observations=next_states,
        )


class RolloutCollector():
    def __init__(self, env, policy_model, n_steps, stats_window_size=100,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 normalize_advantage: bool = True, advantages_norm_eps: float = 1e-8,
                 use_gae: bool = True,
                 normalize_advantages: bool = False,
                 buffer_maxsize: Optional[int] = None,
                 **kwargs):
        self.env = env
        self.policy_model = policy_model
        self.n_steps = n_steps
        self.stats_window_size = stats_window_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.advantages_norm_eps = advantages_norm_eps
        self.use_gae = use_gae
        # Keep both flags for backward-compat; use normalize_advantages in logic
        self.normalize_advantage = normalize_advantage
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
        self._obs_count = 0
        self._obs_sum = 0.0
        self._obs_sumsq = 0.0

        self._rew_count = 0
        self._rew_sum = 0.0
        self._rew_sumsq = 0.0

        # Action histogram (discrete); grows dynamically as needed
        self._action_counts = None

        # Baseline stats for REINFORCE (global running mean/std over returns)
        self._base_count = 0
        self._base_sum = 0.0
        self._base_sumsq = 0.0

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

    @torch.inference_mode()
    def collect(self, *args, **kwargs):
        with inference_ctx(self.policy_model):
            return self._collect(*args, **kwargs)

    def _collect(self, deterministic=False):
        """Collect a single rollout and return trajectories and stats."""
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
                # Collect episode stats for all done environments
                for idx in done_indices:
                    info = infos[idx]
                    episode = info['episode']
                    self.episode_reward_deque.append(episode['r'])
                    self.episode_length_deque.append(episode['l'])
                    self.env_episode_reward_deques[idx].append(episode['r'])
                    self.env_episode_length_deques[idx].append(episode['l'])

                    # Track immediate last episode stats (latest wins if multiple end simultaneously)
                    try:
                        self._last_episode_reward = float(episode.get('r', 0.0))
                    except Exception:
                        self._last_episode_reward = 0.0
                    try:
                        self._last_episode_length = int(episode.get('l', 0))
                    except Exception:
                        self._last_episode_length = 0

                    # Just mark timeouts and collect terminal obs for later processing
                    if info.get("TimeLimit.truncated"):
                        timeouts[idx] = True
                        terminal_obs_info.append((step_idx, idx, info["terminal_observation"]))

            # Persist environment outputs for this step
            self._buffer.store_cpu_step(
                start + step_idx,
                self.obs.copy(),
                next_obs.copy(),
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
        self._obs_count += obs_vals.size
        self._obs_sum += float(obs_vals.sum())
        self._obs_sumsq += float((obs_vals * obs_vals).sum())

        # Rewards
        rewards_flat = self._buffer.rewards_buf[start:end].ravel()
        self._rew_count += rewards_flat.size
        self._rew_sum += float(rewards_flat.sum())
        self._rew_sumsq += float((rewards_flat * rewards_flat).sum())

        # Actions: maintain a histogram of encountered actions
        actions_flat = self._buffer.actions_buf[start:end].ravel()
        if actions_flat.size:
            amax = int(actions_flat.max())
            if self._action_counts is None:
                self._action_counts = np.zeros(amax + 1, dtype=np.int64)
            elif amax >= self._action_counts.shape[0]:
                # grow counts array
                new_counts = np.zeros(amax + 1, dtype=np.int64)
                new_counts[: self._action_counts.shape[0]] = self._action_counts
                self._action_counts = new_counts
            binc = np.bincount(actions_flat, minlength=self._action_counts.shape[0])
            self._action_counts[: len(binc)] += binc

        # Single batch transfer of GPU tensors to CPU after rollout collection for this slice
        self._buffer.copy_tensors_to_cpu(start, end)

        # Batch process all terminal observations at once (major performance improvement)
        if terminal_obs_info:
            terminal_observations = [info[2] for info in terminal_obs_info]
            term_obs_batch = np.stack(terminal_observations)
            term_obs_t = torch.as_tensor(term_obs_batch, dtype=torch.float32, device=self.device)

            # Single batch prediction for all terminal observations
            batch_values = self.policy_model.predict_values(term_obs_t).detach().cpu().numpy().squeeze().astype(np.float32)

            # Handle single vs multiple predictions
            if len(terminal_obs_info) == 1:
                batch_values = [batch_values]

            # Assign batch results back to correct buffer positions
            for i, (step_idx_term, env_idx, _) in enumerate(terminal_obs_info):
                self._buffer.bootstrapped_values_buf[start + step_idx_term, env_idx] = batch_values[i]

        # Build next_values by shifting critic values and estimating the last
        values_slice = self._buffer.values_buf[start:end]
        rewards_slice = self._buffer.rewards_buf[start:end]
        dones_slice = self._buffer.dones_buf[start:end]
        timeouts_slice = self._buffer.timeouts_buf[start:end]

        next_values_buf = np.zeros_like(values_slice, dtype=np.float32)
        next_values_buf[:-1] = values_slice[1:]
        last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
        last_values = self.policy_model.predict_values(last_obs_t).detach().cpu().numpy().squeeze().astype(np.float32)
        next_values_buf[-1] = last_values

        # Override next values array with bootstrapped values from terminal states for truncated episodes
        bootstrapped_slice = self._buffer.bootstrapped_values_buf[start:end]
        next_values_buf = np.where(timeouts_slice, bootstrapped_slice, next_values_buf)

        # Real terminal states are only the dones where environment finished but not due to a timeout
        real_terminal = np.logical_and(dones_slice.astype(bool), ~timeouts_slice)
        non_terminal = (~real_terminal).astype(np.float32)

        if self.use_gae:
            # Calculate the advantages using GAE(λ):
            advantages_buf = np.zeros_like(rewards_slice, dtype=np.float32)
            gae = np.zeros(self.n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                # Calculate the Temporal Difference (TD) residual
                delta = rewards_slice[t] + self.gamma * next_values_buf[t] * non_terminal[t] - values_slice[t]

                # Accumulate GAE
                gae = delta + self.gamma * self.gae_lambda * gae * non_terminal[t]
                advantages_buf[t] = gae

            # For GAE, returns are advantages + value estimates
            returns_buf = advantages_buf + values_slice
        else:
            # Monte Carlo returns for REINFORCE
            returns_buf = np.zeros_like(rewards_slice, dtype=np.float32)
            returns = np.zeros(self.n_envs, dtype=np.float32)

            for t in reversed(range(T)):
                # For terminal states, return is just the reward; for non-terminal, accumulate discounted future returns
                returns = rewards_slice[t] + self.gamma * returns * non_terminal[t]
                returns_buf[t] = returns

            # Always calculate baseline and advantages using global running mean
            returns_flat = returns_buf.ravel()
            # Update baseline running stats (global over samples)
            self._base_count += returns_flat.size
            self._base_sum += float(returns_flat.sum())
            self._base_sumsq += float((returns_flat * returns_flat).sum())

            if self._base_count > 0:
                baseline = self._base_sum / self._base_count
                advantages_buf = returns_buf - baseline
            else:
                advantages_buf = returns_buf.copy()

        if self.normalize_advantages:
            adv_flat = advantages_buf.reshape(-1)
            advantages_buf = (advantages_buf - adv_flat.mean()) / (adv_flat.std() + self.advantages_norm_eps)

        # Create final training tensors from persistent buffers
        trajectories = self._buffer.flatten_slice_env_major(start, end, advantages_buf, returns_buf)

        self.total_rollouts += 1
        rollout_elapsed = time.time() - rollout_start
        fps = len(trajectories.observations) / rollout_elapsed  # steps per second
        self.rollout_fpss.append(fps)

        return trajectories

    def slice_trajectories(self, trajectories, idxs):
        return RolloutTrajectory(
            observations=trajectories.observations[idxs],
            actions=trajectories.actions[idxs],
            rewards=trajectories.rewards[idxs],
            dones=trajectories.dones[idxs],
            old_log_prob=trajectories.old_log_prob[idxs],  # TODO: change names
            old_values=trajectories.old_values[idxs],
            advantages=trajectories.advantages[idxs],
            returns=trajectories.returns[idxs],
            next_observations=trajectories.next_observations[idxs]
        )

    def get_metrics(self):
        ep_rew_mean = float(self.episode_reward_deque.mean()) if self.episode_reward_deque else 0.0
        ep_len_mean = int(self.episode_length_deque.mean()) if self.episode_length_deque else 0
        rollout_fps = float(self.rollout_fpss.mean()) if self.rollout_fpss else 0.0

        # Observation statistics from running aggregates
        if self._obs_count > 0:
            obs_mean = self._obs_sum / self._obs_count
            obs_var = max(0.0, self._obs_sumsq / self._obs_count - obs_mean * obs_mean)
            obs_std = float(np.sqrt(obs_var))
        else:
            obs_mean = 0.0
            obs_std = 0.0

        # Reward statistics
        if self._rew_count > 0:
            reward_mean = self._rew_sum / self._rew_count
            reward_var = max(0.0, self._rew_sumsq / self._rew_count - reward_mean * reward_mean)
            reward_std = float(np.sqrt(reward_var))
        else:
            reward_mean = 0.0
            reward_std = 0.0

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
        if self._base_count > 0:
            baseline_mean = self._base_sum / self._base_count
            base_var = max(0.0, self._base_sumsq / self._base_count - baseline_mean * baseline_mean)
            baseline_std = float(np.sqrt(base_var))
        else:
            baseline_mean, baseline_std = 0.0, 0.0

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
        if counts is None:
            return None
        out = counts.copy()
        if reset:
            self._action_counts = np.zeros_like(counts)
        return out
