from collections import deque
from typing import NamedTuple, Deque, Tuple, Optional

import time
import numpy as np
import torch

from utils.misc import inference_ctx, _device_of, calculate_deque_stats

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

class RolloutBuffer:
    """
    Persistent rollout storage to avoid reallocating buffers every collect.

    - Preallocates CPU and GPU buffers of size (maxsize, n_envs, ...)
    - begin_rollout(T) ensures a contiguous slice is available; if not, resets
      write position to 0 so the rollout fits contiguously.
    - Collector stores per-step data using store_* methods.
    - Provides utilities to copy GPU->CPU for a slice and to flatten a slice
      into env-major tensors suitable for training.
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

        # GPU buffers
        self.obs_tensor_buf = torch.zeros((self.maxsize, self.n_envs, *self.obs_shape), dtype=torch.float32, device=self.device)
        self.actions_tensor_buf = torch.zeros((self.maxsize, self.n_envs), dtype=torch.int64, device=self.device)
        self.logprobs_tensor_buf = torch.zeros((self.maxsize, self.n_envs), dtype=torch.float32, device=self.device)
        self.values_tensor_buf = torch.zeros((self.maxsize, self.n_envs), dtype=torch.float32, device=self.device)

        # write position tracking
        self.pos: int = 0
        self.size: int = 0  # number of valid steps currently stored (for info)

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
        self.obs_tensor_buf[idx] = obs_t
        self.actions_tensor_buf[idx] = actions_t
        self.logprobs_tensor_buf[idx] = logps_t
        self.values_tensor_buf[idx] = values_t

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
        self.obs_buf[idx] = obs_np
        self.next_obs_buf[idx] = next_obs_np
        self.actions_buf[idx] = actions_np
        self.rewards_buf[idx] = rewards_np
        self.dones_buf[idx] = dones_np
        self.timeouts_buf[idx] = timeouts_np

    def copy_tensors_to_cpu(self, start: int, end: int) -> None:
        """Batch transfer GPU tensors to CPU numpy buffers for [start, end)."""
        self.logprobs_buf[start:end] = self.logprobs_tensor_buf[start:end].detach().cpu().numpy()
        self.values_buf[start:end] = self.values_tensor_buf[start:end].detach().cpu().numpy()

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

        # Observations already on GPU
        obs_slice = self.obs_tensor_buf[start:end]
        states = obs_slice.transpose(0, 1).reshape(n_envs * T, -1)

        def _flat_env_major_gpu(tensor_buf: torch.Tensor) -> torch.Tensor:
            return tensor_buf[start:end].transpose(0, 1).reshape(-1)

        actions = _flat_env_major_gpu(self.actions_tensor_buf).to(torch.int64)
        logps = _flat_env_major_gpu(self.logprobs_tensor_buf).to(torch.float32)
        values = _flat_env_major_gpu(self.values_tensor_buf).to(torch.float32)

        def _flat_env_major_cpu(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(arr[start:end].transpose(1, 0).reshape(-1), dtype=dtype, device=self.device)

        rewards = _flat_env_major_cpu(self.rewards_buf, torch.float32)
        dones = _flat_env_major_cpu(self.dones_buf, torch.bool)
        advantages = _flat_env_major_cpu(advantages_buf, torch.float32)
        returns = _flat_env_major_cpu(returns_buf, torch.float32)

        # Next observations
        if self.next_obs_buf.ndim == 2:
            next_states = torch.as_tensor(
                self.next_obs_buf[start:end].transpose(1, 0).reshape(-1),
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
        self.normalize_advantage = normalize_advantage
        self.advantages_norm_eps = advantages_norm_eps
        self.use_gae = use_gae
        self.normalize_advantages = normalize_advantages
        self.kwargs = kwargs
        self.buffer_maxsize = buffer_maxsize

        # State tracking
        self.device = _device_of(policy_model)
        self.n_envs = env.num_envs

        # Running average stats (windowed)
        self.rollout_fpss = deque(maxlen=stats_window_size)
        self.episode_reward_deque = deque(maxlen=stats_window_size)
        self.episode_length_deque = deque(maxlen=stats_window_size)
        self.env_episode_reward_deques = [deque(maxlen=stats_window_size) for _ in range(self.n_envs)]
        self.env_episode_length_deques = [deque(maxlen=stats_window_size) for _ in range(self.n_envs)]

        # Observation and reward statistics tracking
        self.obs_values_deque = deque(maxlen=stats_window_size)
        self.reward_values_deque = deque(maxlen=stats_window_size)

        # Action statistics tracking
        self.action_values_deque = deque(maxlen=stats_window_size)

        # Baseline tracking for REINFORCE (rolling average of returns)
        self.baseline_deque = deque(maxlen=stats_window_size)

        self.total_rollouts = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.rollout_steps = 0
        self.rollout_episodes = 0

        # Current observations - initialize on first collect
        self.obs = None

        # Persistent rollout buffer (lazy init when first obs is known)
        self._buffer = None

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
            obs_shape = self.obs.shape[1:] if self.obs.ndim > 1 else (self.obs.shape[0],)
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
            # Store observations directly in GPU tensor buffer
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)

            # Determine next actions using the policy model (already on GPU)
            actions_t, logps_t, values_t = self.policy_model.act(obs_t, deterministic=deterministic)

            # Store GPU tensors directly in persistent buffers
            self._buffer.store_tensors(start + step_idx, obs_t, actions_t, logps_t, values_t)

            # Only transfer actions to CPU for environment step
            actions_np = actions_t.detach().cpu().numpy()

            # Perform next actions on environment
            next_obs, rewards, dones, infos = self.env.step(actions_np)

            # Fast episode info processing - just collect data, delay computation
            timeouts = np.zeros(self.n_envs, dtype=bool)

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

                    # Just mark timeouts and collect terminal obs for later processing
                    if info.get("TimeLimit.truncated"):
                        timeouts[idx] = True
                        terminal_obs_info.append((step_idx, idx, info["terminal_observation"]))

            # Direct CPU buffer writes (persistent)
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

        # Collect observation and reward statistics from this rollout
        # Flatten observations and rewards across time and environments for statistics
        obs_flat = self._buffer.obs_buf[start:end].reshape(-1, *self._buffer.obs_buf.shape[2:])
        rewards_flat = self._buffer.rewards_buf[start:end].flatten()
        actions_flat = self._buffer.actions_buf[start:end].flatten()

        # Store flattened observations, rewards, and actions for windowed statistics
        self.obs_values_deque.extend(obs_flat)
        self.reward_values_deque.extend(rewards_flat)
        self.action_values_deque.extend(actions_flat)

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

        # Create a next values array to use for GAE(λ), by shifting the critic
        # values on steps (discards first value estimation, which is not used),
        # and estimating the last value using the value model)
        values_slice = self._buffer.values_buf[start:end]
        next_values_buf = np.zeros_like(values_slice, dtype=np.float32)
        next_values_buf[:-1] = values_slice[1:]
        last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
        last_values = self.policy_model.predict_values(last_obs_t).detach().cpu().numpy().squeeze().astype(np.float32)
        next_values_buf[-1] = last_values

        # Override next values array with bootstrapped values from terminal states for truncated episodes
        # - by default last obs is next state from unfinished episode
        # - if episode is done and truncated, value will be replaced by value from terminal observation (as last obs here will be the next states' first observation)
        # - if episode is done but not truncated, value will later be ignored in GAE calculation by this being considered a terminal state
        timeouts_slice = self._buffer.timeouts_buf[start:end]
        bootstrapped_slice = self._buffer.bootstrapped_values_buf[start:end]
        next_values_buf = np.where(timeouts_slice, bootstrapped_slice, next_values_buf)

        # Real terminal states are only the dones where environment finished but not due to a timeout
        # (for timeout we must estimate next state value as if episode continued)
        dones_slice = self._buffer.dones_buf[start:end]
        real_terminal = np.logical_and(dones_slice.astype(bool), ~timeouts_slice)
        non_terminal = (~real_terminal).astype(np.float32)

        if self.use_gae:
            # Calculate the advantages using GAE(λ):
            advantages_buf = np.zeros_like(self._buffer.rewards_buf[start:end], dtype=np.float32)
            gae = np.zeros(self.n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                # Calculate the Temporal Difference (TD) residual (the error
                # between the predicted value of a state and a better estimate of it)
                delta = self._buffer.rewards_buf[start:end][t] + self.gamma * next_values_buf[t] * non_terminal[t] - values_slice[t]

                # The TD residual is a 1-step advantage estimate, by taking future advantage
                # estimates into account the advantage estimate becomes more stable
                gae = delta + self.gamma * self.gae_lambda * gae * non_terminal[t]
                advantages_buf[t] = gae

            # For GAE, returns are advantages + value estimates
            returns_buf = advantages_buf + values_slice
        else:
            # Monte Carlo returns for REINFORCE
            returns_buf = np.zeros_like(self._buffer.rewards_buf[start:end], dtype=np.float32)
            returns = np.zeros(self.n_envs, dtype=np.float32)

            for t in reversed(range(T)):
                # For terminal states, return is just the reward; for non-terminal, accumulate discounted future returns
                returns = self._buffer.rewards_buf[start:end][t] + self.gamma * returns * non_terminal[t]
                returns_buf[t] = returns

            # Always calculate baseline (rolling average of returns) and advantages
            # Flatten returns across time and environments for baseline update
            returns_flat = returns_buf.flatten()
            self.baseline_deque.extend(returns_flat)

            # Calculate baseline from rolling average
            if len(self.baseline_deque) > 0:
                baseline = float(np.mean(self.baseline_deque))
                advantages_buf = returns_buf - baseline
            else:
                # If no baseline history yet, advantages equal returns (no baseline subtraction)
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

    def get_metrics(self):
        ep_rew_mean = float(np.mean(self.episode_reward_deque)) if self.episode_reward_deque else 0.0
        ep_len_mean = int(np.mean(self.episode_length_deque)) if self.episode_length_deque else 0
        rollout_fps = float(np.mean(self.rollout_fpss)) if self.rollout_fpss else 0.0

        # Calculate observation statistics
        obs_mean, obs_std, _ = calculate_deque_stats(self.obs_values_deque)
        
        # Calculate reward statistics
        reward_mean, reward_std, _ = calculate_deque_stats(self.reward_values_deque)

        # Calculate action statistics
        action_mean, action_std, action_distribution = calculate_deque_stats(
            self.action_values_deque, return_distribution=True
        )

        # Calculate baseline statistics (always calculated now)
        baseline_mean, baseline_std, _ = calculate_deque_stats(self.baseline_deque)

        return {
            "total_timesteps": self.total_steps, # TODO: steps vs timesteps
            "total_episodes": self.total_episodes,
            "total_rollouts": self.total_rollouts,
            "rollout_timesteps": self.rollout_steps,
            "rollout_episodes": self.rollout_episodes,  # Renamed to avoid conflict with video logging
            "rollout_fps": rollout_fps, # TODO: this is a mean, it shouln't be
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "action_distribution": action_distribution,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std
        }
