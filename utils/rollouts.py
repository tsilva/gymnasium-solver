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
    *,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = True,
    advantages_norm_eps: float = 1e-8,
    stats_window_size: int = 100
):
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    device = _device_of(policy_model)
    #device: torch.device = policy_device

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
        # Reset rollout buffers for one collection window
        # --------------------------------------------------------------
        obs_buf: list[np.ndarray] = []
        actions_buf: list[np.ndarray] = []
        rewards_buf: list[np.ndarray] = []
        values_buf: list[np.ndarray] = []
        done_buf: list[np.ndarray] = []
        logprobs_buf: list[np.ndarray] = []
        timeouts_buf: list[np.ndarray] = []
        bootstrapped_values_buf: list[np.ndarray] = []

        env_step_calls = 0
        rollout_steps = 0
        rollout_episodes = 0

        # --------------------------------------------------------------
        # Collect one rollout
        # --------------------------------------------------------------
        with inference_ctx(policy_model):
            rollout_start = time.time()
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

                # Buffer writes
                obs_buf.append(obs.copy())  # defensive copy — some envs mutate obs
                actions_buf.append(actions_np)
                logprobs_buf.append(action_logp_np)
                rewards_buf.append(rewards)
                values_buf.append(value_np)
                done_buf.append(dones)
                timeouts_buf.append(timeouts)
                bootstrapped_values_buf.append(bootstrapped_values)

                # Advance
                obs = next_obs
                env_step_calls += 1
                rollout_steps += n_envs
                rollout_episodes += int(dones.sum())

                # Stop conditions
                if n_steps is not None and env_step_calls >= n_steps: break
                if n_episodes is not None and rollout_episodes >= n_episodes: break

            last_obs = next_obs
            total_steps += rollout_steps
            total_episodes += rollout_episodes

            # TODO: can I operate directly on numpy arrays?
            # ----- stack arrays -----
            obs_arr = np.stack(obs_buf)            # (T, E, ...)
            actions_arr = np.stack(actions_buf)    # (T, E, ...) or (T, E)
            logprobs_arr = np.stack(logprobs_buf)  # (T, E)
            rewards_arr = np.stack(rewards_buf)    # (T, E)
            values_arr = np.stack(values_buf)      # (T, E)
            dones_arr = np.stack(done_buf)         # (T, E)
            timeouts_arr = np.stack(timeouts_buf)  # (T, E) bool
            bootstrap_values_arr = np.stack(bootstrapped_values_buf)  # (T, E) float32
            T = obs_arr.shape[0]

            # TODO: consider just replacing the terminal states in last obs
            # Create a next values array to use for GAE(λ), by shifting the critic 
            # values on steps (discards first value estimation, which is not used), 
            # and estimating the last value using the value model)
            next_values_arr = np.zeros_like(values_arr, dtype=np.float32)
            next_values_arr[:-1] = values_arr[1:]
            last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=device)
            last_values = policy_model.predict_values(last_obs_t).detach().cpu().numpy().squeeze()
            next_values_arr[-1] = last_values.astype(np.float32)

            # Override next values array with boostrapped values from terminal states for truncated episodes
            # - by default last obs is next state from unfinished episode
            # - if episode is done and truncated, value will be replaced by value from terminal observation (as last obs here will be the next states' first observation)
            # - if episode is done but not truncated, value will later be ignored in GAE calculation by this being considered a terminal state
            next_values_arr = np.where(timeouts_arr, bootstrap_values_arr, next_values_arr)

            # Real terminal states are only the dones where environment finished but not due to a timeout
            # (for timeout we must estimate next state value as if episode continued)
            real_terminal = np.logical_and(dones_arr.astype(bool), ~timeouts_arr)
            non_terminal = (~real_terminal).astype(np.float32)

            # Calculate the advantages using GAE(λ):
            advantages_arr = np.zeros_like(rewards_arr, dtype=np.float32)
            gae = np.zeros(n_envs, dtype=np.float32)
            for t in reversed(range(T)):
                # Calculate the Temporal Difference (TD) residual (the error 
                # between the predicted value of a state and a better estimate of it)
                delta = rewards_arr[t] + gamma * next_values_arr[t] * non_terminal[t] - values_arr[t]

                # The TD residual is a 1-step advantage estimate, by taking future advantage 
                # estimates into account the advantage estimate becomes more stable
                gae = delta + gamma * gae_lambda * gae * non_terminal[t]
                advantages_arr[t] = gae
            
            # TODO: consider calculating in loop and then asserting same
            returns_arr = advantages_arr + values_arr

            # Normalize advantages across rollout (we could normalize across training batches 
            # later on, but in some situations normalizing across rollouts provides better numerical stability)
            if normalize_advantage:
                adv_flat = advantages_arr.reshape(-1)
                advantages_arr = (advantages_arr - adv_flat.mean()) / (adv_flat.std() + advantages_norm_eps)

            # TODO: get rid of this code if possible
            obs_env_major = np.swapaxes(obs_arr, 0, 1)
            states = torch.as_tensor(obs_env_major.reshape(n_envs * T, -1), dtype=torch.float32)
            def _flat_env_major(arr: np.ndarray, dtype: torch.dtype): return torch.as_tensor(arr.transpose(1, 0).reshape(-1), dtype=dtype)
            actions = _flat_env_major(actions_arr, torch.int64)
            rewards = _flat_env_major(rewards_arr, torch.float32)
            dones = _flat_env_major(dones_arr, torch.bool)
            logps = _flat_env_major(logprobs_arr, torch.float32)
            values = _flat_env_major(values_arr, torch.float32)
            advantages = _flat_env_major(advantages_arr, torch.float32)
            returns = _flat_env_major(returns_arr, torch.float32)

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
    def __init__(self, env, policy_model, n_steps=None, n_episodes=None, stats_window_size=100, **kwargs):
        self.env = env
        self.policy_model = policy_model
        self.n_steps = n_steps
        self.n_episodes = n_episodes
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
    
    def _ensure_generator(self, n_episodes=None, n_steps=None):
        if self._generator: return self._generator

        n_episodes = n_episodes if n_episodes is not None else self.n_episodes
        n_steps = n_steps if n_steps is not None else self.n_steps
        self._generator = _collect_rollouts( # TODO: don't return tensors, return numpy arrays?
            self.env,
            self.policy_model,
            n_steps=n_steps,
            n_episodes=n_episodes,
            stats_window_size=self.stats_window_size,
            **self.kwargs
        )
        return self._generator
    
    def __del__(self):
        if self._generator is None: return 
        self._generator.close() # TODO: drop generator pattern, merge code into class
        self.env.close() # TODO: is this the responsibility of rollout collector or whoever created it?
