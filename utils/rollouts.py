from collections import deque
from typing import NamedTuple, Deque

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from utils.misc import inference_ctx, _device_of

class RolloutTrajectory(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RolloutSample(RolloutTrajectory):
    pass

class RolloutDataset(TorchDataset):
    def __init__(self): # TODO: make sure dataset is not being tampered with during collection
        self.trajectories = None

    def update(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        length = len(self.trajectories.observations)
        return length

    def __getitem__(self, idx):
        return RolloutSample(
            observations=self.trajectories.observations[idx],
            actions=self.trajectories.actions[idx],
            rewards=self.trajectories.rewards[idx],
            dones=self.trajectories.dones[idx],
            old_log_prob=self.trajectories.old_log_prob[idx],
            old_values=self.trajectories.old_values[idx],
            advantages=self.trajectories.advantages[idx],
            returns=self.trajectories.returns[idx]
        )
    
# NOTE: Don't perform changes that result in CartPole-v1 with PPO being solvable in more than 100096 steps (around 16 secs)
class RolloutCollector():
    def __init__(self, env, policy_model, n_steps, stats_window_size=100, 
                 gamma: float = 0.99, gae_lambda: float = 0.95, 
                 normalize_advantage: bool = True, advantages_norm_eps: float = 1e-8, 
                 use_gae: bool = True,
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
        self.kwargs = kwargs
        
        # State tracking
        self.device = _device_of(policy_model)
        self.n_envs: int = env.num_envs
        
        # Running average stats (windowed)
        self.rollout_durations: Deque[float] = deque(maxlen=stats_window_size)
        self.episode_reward_deque: Deque[float] = deque(maxlen=stats_window_size)
        self.episode_length_deque: Deque[int] = deque(maxlen=stats_window_size)
        self.env_episode_reward_deques = [deque(maxlen=stats_window_size) for _ in range(self.n_envs)]
        self.env_episode_length_deques = [deque(maxlen=stats_window_size) for _ in range(self.n_envs)]
        
        # Observation and reward statistics tracking
        self.obs_values_deque: Deque[np.ndarray] = deque(maxlen=stats_window_size)
        self.reward_values_deque: Deque[float] = deque(maxlen=stats_window_size)
        
        # Action statistics tracking
        self.action_values_deque: Deque[int] = deque(maxlen=stats_window_size)
        
        self.total_rollouts: int = 0
        self.total_steps: int = 0
        self.total_episodes: int = 0
        self.rollout_steps: int = 0
        self.rollout_episode: int = 0
        
        self.dataset = RolloutDataset()

        # Current observations - initialize on first collect
        self.obs = None

    @torch.inference_mode()
    def collect(self, *args, **kwargs):
        with inference_ctx(self.policy_model): 
            return self._collect(*args, **kwargs)

    def _collect(self, deterministic=False):
        """Collect a single rollout and return trajectories and stats."""
        # Initialize environment if needed
        if self.obs is None:
            self.obs = self.env.reset()

        # --------------------------------------------------------------
        # Pre-allocate rollout buffers for one collection window
        # --------------------------------------------------------------
        buffer_size = self.n_steps
        obs_shape = self.obs.shape[1:] if self.obs.ndim > 1 else (self.obs.shape[0],)
        
        obs_buf = np.zeros((buffer_size, self.n_envs, *obs_shape), dtype=self.obs.dtype)
        actions_buf = np.zeros((buffer_size, self.n_envs), dtype=np.int64)
        rewards_buf = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        values_buf = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        dones_buf = np.zeros((buffer_size, self.n_envs), dtype=bool)
        logprobs_buf = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        timeouts_buf = np.zeros((buffer_size, self.n_envs), dtype=bool)
        bootstrapped_values_buf = np.zeros((buffer_size, self.n_envs), dtype=np.float32)
        
        # Pre-allocate GPU tensors to minimize transfers
        obs_tensor_buf = torch.zeros((buffer_size, self.n_envs, *obs_shape), dtype=torch.float32, device=self.device)
        actions_tensor_buf = torch.zeros((buffer_size, self.n_envs), dtype=torch.int64, device=self.device)
        logprobs_tensor_buf = torch.zeros((buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        values_tensor_buf = torch.zeros((buffer_size, self.n_envs), dtype=torch.float32, device=self.device)
        
        # Collect terminal observations for later batch processing
        terminal_obs_info = []  # List of (step_idx, env_idx, terminal_obs)

        env_step_calls = 0
        self.rollout_steps = 0
        self.rollout_episodes = 0

        # --------------------------------------------------------------
        # Collect one rollout
        # --------------------------------------------------------------
        rollout_start = time.time()
        step_idx = 0
        while env_step_calls < self.n_steps:
            # Store observations directly in GPU tensor buffer
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
            obs_tensor_buf[step_idx] = obs_t
            
            # Determine next actions using the policy model (already on GPU)
            actions_t, logps_t, values_t = self.policy_model.act(obs_t, deterministic=deterministic)
            
            # Store GPU tensors directly in pre-allocated buffers
            actions_tensor_buf[step_idx] = actions_t
            logprobs_tensor_buf[step_idx] = logps_t  
            values_tensor_buf[step_idx] = values_t
            
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

            # Direct buffer writes
            obs_buf[step_idx] = self.obs.copy()  # defensive copy — some envs mutate obs
            actions_buf[step_idx] = actions_np
            rewards_buf[step_idx] = rewards
            dones_buf[step_idx] = dones
            timeouts_buf[step_idx] = timeouts
            # Note: logprobs_buf and values_buf will be filled from GPU tensors later
            # bootstrapped_values_buf will be filled after rollout collection

            # Advance
            self.obs = next_obs
            env_step_calls += 1
            self.rollout_steps += self.n_envs
            self.rollout_episodes += int(dones.sum())
            step_idx += 1

        last_obs = next_obs
        self.total_steps += self.rollout_steps
        self.total_episodes += self.rollout_episodes

        # Use buffers directly since they're pre-allocated to exact size
        T = step_idx
        
        # Collect observation and reward statistics from this rollout
        # Flatten observations and rewards across time and environments for statistics
        obs_flat = obs_buf[:T].reshape(-1, *obs_buf.shape[2:])  # Shape: (T*n_envs, *obs_shape)
        rewards_flat = rewards_buf[:T].flatten()  # Shape: (T*n_envs,)
        actions_flat = actions_buf[:T].flatten()  # Shape: (T*n_envs,)
        
        # Store flattened observations, rewards, and actions for windowed statistics
        self.obs_values_deque.extend(obs_flat)
        self.reward_values_deque.extend(rewards_flat)
        self.action_values_deque.extend(actions_flat)
        
        # Single batch transfer of GPU tensors to CPU after rollout collection
        logprobs_buf[:T] = logprobs_tensor_buf[:T].detach().cpu().numpy()
        values_buf[:T] = values_tensor_buf[:T].detach().cpu().numpy()

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
                bootstrapped_values_buf[step_idx_term, env_idx] = batch_values[i]

        # Create a next values array to use for GAE(λ), by shifting the critic 
        # values on steps (discards first value estimation, which is not used), 
        # and estimating the last value using the value model)
        next_values_buf = np.zeros_like(values_buf, dtype=np.float32)
        next_values_buf[:-1] = values_buf[1:]
        last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
        last_values = self.policy_model.predict_values(last_obs_t).detach().cpu().numpy().squeeze().astype(np.float32)
        next_values_buf[-1] = last_values

        # Override next values array with boostrapped values from terminal states for truncated episodes
        # - by default last obs is next state from unfinished episode
        # - if episode is done and truncated, value will be replaced by value from terminal observation (as last obs here will be the next states' first observation)
        # - if episode is done but not truncated, value will later be ignored in GAE calculation by this being considered a terminal state
        next_values_buf = np.where(timeouts_buf, bootstrapped_values_buf, next_values_buf)

        # Real terminal states are only the dones where environment finished but not due to a timeout
        # (for timeout we must estimate next state value as if episode continued)
        real_terminal = np.logical_and(dones_buf.astype(bool), ~timeouts_buf)
        non_terminal = (~real_terminal).astype(np.float32)

        if self.use_gae:
            # Calculate the advantages using GAE(λ):
            advantages_buf = np.zeros_like(rewards_buf, dtype=np.float32)
            gae = np.zeros(self.n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                # Calculate the Temporal Difference (TD) residual (the error 
                # between the predicted value of a state and a better estimate of it)
                delta = rewards_buf[t] + self.gamma * next_values_buf[t] * non_terminal[t] - values_buf[t]

                # The TD residual is a 1-step advantage estimate, by taking future advantage 
                # estimates into account the advantage estimate becomes more stable
                gae = delta + self.gamma * self.gae_lambda * gae * non_terminal[t]
                advantages_buf[t] = gae
            
            # For GAE, returns are advantages + value estimates
            returns_buf = advantages_buf + values_buf
        else:
            # Monte Carlo returns for REINFORCE
            returns_buf = np.zeros_like(rewards_buf, dtype=np.float32)
            returns = np.zeros(self.n_envs, dtype=np.float32)
            
            for t in reversed(range(T)):
                # For terminal states, return is just the reward; for non-terminal, accumulate discounted future returns
                returns = rewards_buf[t] + self.gamma * returns * non_terminal[t]
                returns_buf[t] = returns
            
            # For REINFORCE, advantages are the returns themselves (no baseline subtraction)
            advantages_buf = returns_buf.copy()

        # Normalize advantages across rollout (we could normalize across training batches 
        # later on, but in some situations normalizing across rollouts provides better numerical stability)
        if self.normalize_advantage:
            # TODO: create normalization util
            adv_flat = advantages_buf.reshape(-1)
            advantages_buf = (advantages_buf - adv_flat.mean()) / (adv_flat.std() + self.advantages_norm_eps)

        # Create final tensors for training - minimize CPU-GPU transfers
        # Use pre-allocated GPU buffers and reshape efficiently
        obs_env_major = obs_tensor_buf[:T].transpose(0, 1).reshape(self.n_envs * T, -1)
        states = obs_env_major  # Already on GPU as float32
        
        def _flat_env_major_gpu(tensor_buf: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
            return tensor_buf[:T].transpose(0, 1).reshape(-1).to(dtype)
        
        actions = _flat_env_major_gpu(actions_tensor_buf, T, torch.int64)
        logps = _flat_env_major_gpu(logprobs_tensor_buf, T, torch.float32)
        values = _flat_env_major_gpu(values_tensor_buf, T, torch.float32)
        
        # For CPU-computed arrays, do single batch transfers
        def _flat_env_major_cpu(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(arr[:T].transpose(1, 0).reshape(-1), dtype=dtype, device=self.device)
        
        rewards = _flat_env_major_cpu(rewards_buf, torch.float32)
        dones = _flat_env_major_cpu(dones_buf, torch.bool)
        advantages = _flat_env_major_cpu(advantages_buf, torch.float32)
        returns = _flat_env_major_cpu(returns_buf, torch.float32)

        # Create trajectories
        trajectories = RolloutTrajectory(
            observations=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            old_log_prob=logps, # TODO: change names
            old_values=values,
            advantages=advantages,
            returns=returns
        )

        self.total_rollouts += 1
        rollout_elapsed = time.time() - rollout_start
        self.rollout_durations.append(rollout_elapsed)

        # TODO: do I need to copy the tuple?
        self.dataset.update(trajectories)

        return trajectories
        
    def create_dataloader(self, batch_size=None, shuffle=None):
        assert batch_size is not None, "batch_size must be specified for create_dataloader"
        assert shuffle is not None, "shuffle must be specified for create_dataloader"
        return DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )

    def get_metrics(self):
        ep_rew_mean = float(np.mean(self.episode_reward_deque)) if self.episode_reward_deque else 0.0
        ep_len_mean = int(np.mean(self.episode_length_deque)) if self.episode_length_deque else 0
        elapsed_mean = float(np.mean(self.rollout_durations))

        # Calculate observation statistics
        obs_mean = obs_std = 0.0
        if self.obs_values_deque:
            obs_array = np.array(list(self.obs_values_deque))
            obs_mean = float(np.mean(obs_array))
            obs_std = float(np.std(obs_array))
        
        # Calculate reward statistics
        reward_mean = reward_std = 0.0
        if self.reward_values_deque:
            reward_array = np.array(list(self.reward_values_deque))
            reward_mean = float(np.mean(reward_array))
            reward_std = float(np.std(reward_array))

        # Calculate action statistics
        action_mean = action_std = 0.0
        action_distribution = None
        if self.action_values_deque:
            action_array = np.array(list(self.action_values_deque))
            action_mean = float(np.mean(action_array))
            action_std = float(np.std(action_array))
            # Store the full action distribution for histogram logging
            action_distribution = action_array

        return {
            "total_timesteps": self.total_steps, # TODO: steps vs timesteps
            "total_episodes": self.total_episodes,
            "total_rollouts": self.total_rollouts,
            "steps": self.rollout_steps,
            "episodes_count": self.rollout_episodes,  # Renamed to avoid conflict with video logging
            "elapsed_mean": elapsed_mean, # TODO: better name?
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "action_distribution": action_distribution
        }
