from collections import deque
from typing import Optional, Sequence

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from utils.misc import inference_ctx, _device_of

# TODO: make rollout collector use its own buffer
# TODO: add # env.normalize_obs() support
# TODO: add # env.normalize_rewards() support
# TODO: batch value inference post rollout
# TODO: consider returning transition object
# TODO: log more stats
# TODO: rename var names
# TODO: look for bugs
# TODO: empirically torch.inference_mode() is not faster than torch.no_grad() for this use case, retest for other envs
# TODO: move to class
import time
from collections import deque
from typing import Optional, Sequence, Deque

import numpy as np
import torch


# TODO: move to class
@torch.no_grad()
def _collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    n_steps,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = True,
    advantages_norm_eps: float = 1e-8,
    stats_window_size: int = 100
):
    device = _device_of(policy_model)
    n_envs: int = env.num_envs

    # Running average stats (windowed)
    rollout_durations: Deque[float] = deque(maxlen=stats_window_size)
    episode_reward_deque: Deque[float] = deque(maxlen=stats_window_size)
    episode_length_deque: Deque[int] = deque(maxlen=stats_window_size)

    total_rollouts: int = 0
    total_steps: int = 0
    total_episodes: int = 0

    # Reset environment and retrieve initial observations
    obs = env.reset()

    while True:
        # --------------------------------------------------------------
        # Pre-allocate rollout buffers for one collection window
        # --------------------------------------------------------------
        # Estimate buffer size - use n_steps if available, otherwise use a reasonable default
        buffer_size = n_steps
        obs_shape = obs.shape[1:] if obs.ndim > 1 else (obs.shape[0],)
        
        obs_buf = np.zeros((buffer_size, n_envs, *obs_shape), dtype=obs.dtype)
        actions_buf = np.zeros((buffer_size, n_envs), dtype=np.int64)
        rewards_buf = np.zeros((buffer_size, n_envs), dtype=np.float32)
        values_buf = np.zeros((buffer_size, n_envs), dtype=np.float32)
        dones_buf = np.zeros((buffer_size, n_envs), dtype=bool)
        logprobs_buf = np.zeros((buffer_size, n_envs), dtype=np.float32)
        timeouts_buf = np.zeros((buffer_size, n_envs), dtype=bool)
        bootstrapped_values_buf = np.zeros((buffer_size, n_envs), dtype=np.float32)

        env_step_calls = 0
        rollout_steps = 0
        rollout_episodes = 0

        # --------------------------------------------------------------
        # Collect one rollout
        # --------------------------------------------------------------
        with inference_ctx(policy_model):
            rollout_start = time.time()
            step_idx = 0
            while True:
                # Determine next actions using the policy model
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions_t, logps_t, values_t = policy_model.act(obs_t)
                actions_np = actions_t.detach().cpu().numpy()
                action_logp_np = logps_t.detach().cpu().numpy()
                value_np = values_t.detach().cpu().numpy()

                # Perform next actions on environment
                next_obs, rewards, dones, infos = env.step(actions_np)

                timeouts = np.zeros(n_envs, dtype=bool)
                bootstrapped_values = np.zeros(n_envs, dtype=np.float32)
                for idx, done in enumerate(dones):
                    if not done: continue

                    info = infos[idx]
                    episode = info['episode']
                    episode_reward_deque.append(episode['r'])
                    episode_length_deque.append(episode['l'])

                    truncated = info.get("TimeLimit.truncated")
                    if not truncated: continue

                    timeouts[idx] = True
                    term_obs = info["terminal_observation"]
                    term_obs_t = torch.as_tensor(term_obs, dtype=torch.float32, device=device)
                    if term_obs_t.ndim == obs_t.ndim - 1: term_obs_t = term_obs_t.unsqueeze(0)  # ensure batch dim
                    bootstrapped_values[idx] = (
                        policy_model.predict_values(term_obs_t)
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                        .astype(np.float32)
                    )

                # Direct buffer writes
                obs_buf[step_idx] = obs.copy()  # defensive copy — some envs mutate obs
                actions_buf[step_idx] = actions_np
                logprobs_buf[step_idx] = action_logp_np
                rewards_buf[step_idx] = rewards
                values_buf[step_idx] = value_np
                dones_buf[step_idx] = dones
                timeouts_buf[step_idx] = timeouts
                bootstrapped_values_buf[step_idx] = bootstrapped_values

                # Advance
                obs = next_obs
                env_step_calls += 1
                rollout_steps += n_envs
                rollout_episodes += int(dones.sum())
                step_idx += 1

                # Stop conditions
                if n_steps is not None and env_step_calls >= n_steps: break

            last_obs = next_obs
            total_steps += rollout_steps
            total_episodes += rollout_episodes

            # Use buffers directly since they're pre-allocated to exact size
            T = step_idx

            # TODO: consider just replacing the terminal states in last obs
            # Create a next values array to use for GAE(λ), by shifting the critic 
            # values on steps (discards first value estimation, which is not used), 
            # and estimating the last value using the value model)
            next_values_buf = np.zeros_like(values_buf, dtype=np.float32)
            next_values_buf[:-1] = values_buf[1:]
            last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=device)
            last_values = policy_model.predict_values(last_obs_t).detach().cpu().numpy().squeeze()
            next_values_buf[-1] = last_values.astype(np.float32)

            # Override next values array with boostrapped values from terminal states for truncated episodes
            # - by default last obs is next state from unfinished episode
            # - if episode is done and truncated, value will be replaced by value from terminal observation (as last obs here will be the next states' first observation)
            # - if episode is done but not truncated, value will later be ignored in GAE calculation by this being considered a terminal state
            next_values_buf = np.where(timeouts_buf, bootstrapped_values_buf, next_values_buf)

            # Real terminal states are only the dones where environment finished but not due to a timeout
            # (for timeout we must estimate next state value as if episode continued)
            real_terminal = np.logical_and(dones_buf.astype(bool), ~timeouts_buf)
            non_terminal = (~real_terminal).astype(np.float32)

            # Calculate the advantages using GAE(λ):
            advantages_buf = np.zeros_like(rewards_buf, dtype=np.float32)
            gae = np.zeros(n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                # Calculate the Temporal Difference (TD) residual (the error 
                # between the predicted value of a state and a better estimate of it)
                delta = rewards_buf[t] + gamma * next_values_buf[t] * non_terminal[t] - values_buf[t]

                # The TD residual is a 1-step advantage estimate, by taking future advantage 
                # estimates into account the advantage estimate becomes more stable
                gae = delta + gamma * gae_lambda * gae * non_terminal[t]
                advantages_buf[t] = gae
            
            # TODO: consider calculating in loop and then asserting same
            returns_buf = advantages_buf + values_buf

            # Normalize advantages across rollout (we could normalize across training batches 
            # later on, but in some situations normalizing across rollouts provides better numerical stability)
            if normalize_advantage:
                adv_flat = advantages_buf.reshape(-1)
                advantages_buf = (advantages_buf - adv_flat.mean()) / (adv_flat.std() + advantages_norm_eps)

            # TODO: get rid of this code if possible
            obs_env_major = np.swapaxes(obs_buf, 0, 1)
            states = torch.as_tensor(obs_env_major.reshape(n_envs * T, -1), dtype=torch.float32)
            def _flat_env_major(arr: np.ndarray, dtype: torch.dtype): return torch.as_tensor(arr.transpose(1, 0).reshape(-1), dtype=dtype)
            actions = _flat_env_major(actions_buf, torch.int64)
            rewards = _flat_env_major(rewards_buf, torch.float32)
            dones = _flat_env_major(dones_buf, torch.bool)
            logps = _flat_env_major(logprobs_buf, torch.float32)
            values = _flat_env_major(values_buf, torch.float32)
            advantages = _flat_env_major(advantages_buf, torch.float32)
            returns = _flat_env_major(returns_buf, torch.float32)

            # ----- Yield -----
            trajectories = (
                states,
                actions,
                rewards,
                dones,
                logps,
                values,
                advantages,
                returns
            )

            # Running means (windowed)
            ep_rew_mean = float(np.mean(episode_reward_deque)) if episode_reward_deque else 0.0
            ep_len_mean = int(np.mean(episode_length_deque)) if episode_length_deque else 0

            total_rollouts += 1
            rollout_elapsed = time.time() - rollout_start
            rollout_durations.append(rollout_elapsed)
            elapsed_mean = float(np.mean(rollout_durations))

            stats = {
                "total_timesteps": total_steps,
                "total_episodes": total_episodes,
                "total_rollouts": total_rollouts,
                "steps": rollout_steps,
                "episodes": rollout_episodes,
                "elapsed_mean": elapsed_mean,
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean,
            }

        yield trajectories, stats

# TODO: add test script for collection using dataset/dataloader
# TODO: should dataloader move to gpu?
# TODO: would converting trajectories to tuples in advance be faster?
class RolloutDataset(TorchDataset):
    def __init__(self, *trajectories): # TODO: make sure dataset is not being tampered with during collection
        self.trajectories = trajectories

    def __len__(self):
        length = len(self.trajectories[0])
        return length

    def __getitem__(self, idx):
        item = tuple(t[idx] for t in self.trajectories)
        return item
    
class RolloutCollector():
    # TODO: how do they perform eval, at which cadence?
    def __init__(self, env, policy_model, n_steps, stats_window_size=100, **kwargs):
        self.env = env
        self.policy_model = policy_model
        self.n_steps = n_steps
        self.stats_window_size = stats_window_size
        self.trajectories = None
        self._generator = None
        self._metrics = {}
        self.kwargs = kwargs

    def collect(self, *args, batch_size=64, shuffle=False, **kwargs):
        generator = self._ensure_generator(*args, **kwargs)
        trajectories, metrics = next(generator)

        self._metrics = metrics
        # NOTE: 
        # - dataloader is created each epoch to mitigate issues with changing dataset data between epochs
        # - multiple workers is not faster because of worker spin up time
        # - peristent workers mitigates worker spin up time, but since dataset data is updated each epoch, workers don't see the updates
        # - therefore, we create a new dataloader each epoch
        return DataLoader(
            RolloutDataset(*trajectories), # TODO: crashes without the *, figure out why
            batch_size=batch_size, 
            shuffle=shuffle
        )

    def get_metrics(self):
        return self._metrics
    
    def get_reward_threshold(self):
        from utils.environment import get_env_reward_threshold
        reward_threshold = get_env_reward_threshold(self.env)
        return reward_threshold
    
    # TODO: merge into get_metric(metric_name)?
    def get_total_timesteps(self):
        return self._metrics.get('total_timesteps', 0)
    
    def get_ep_rew_mean(self):
        return self._metrics.get('ep_rew_mean', 0.0)
    
    def get_total_episodes(self):
        return self._metrics.get('total_episodes', 0)
    
    def is_reward_threshold_reached(self):
        reward_threshold = self.get_reward_threshold()
        ep_rew_mean = self.get_ep_rew_mean()
        total_episodes = self.get_total_episodes()
        reached = total_episodes >= self.stats_window_size and ep_rew_mean >= reward_threshold
        return reached
    
    def _ensure_generator(self):
        if self._generator: return self._generator
        self._generator = _collect_rollouts( # TODO: don't return tensors, return numpy arrays?
            self.env,
            self.policy_model,
            n_steps=self.n_steps,
            stats_window_size=self.stats_window_size,
            **self.kwargs
        )
        return self._generator
    
    def __del__(self):
        if self._generator is None: return 
        self._generator.close() # TODO: drop generator pattern, merge code into class
        self.env.close() # TODO: is this the responsibility of rollout collector or whoever created it?
