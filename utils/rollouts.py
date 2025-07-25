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
@torch.no_grad()
def _collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    *,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    deterministic: bool = False, # TODO: is this still needed? check how rlzoo/sb3 does it
    gamma: float = 0.99, # TODO: make sure these are passed correctly
    gae_lambda: float = 0.95, # TODO: make sure these are passed correctly
    normalize_advantage: bool = True,
    advantages_norm_eps: float = 1e-8,
    collect_frames: bool = False,
    stats_window_size: int = 100,
):
    # ------------------------------------------------------------------
    # 0. Sanity checks --------------------------------------------------
    # ------------------------------------------------------------------
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    policy_device = _device_of(policy_model)

    device: torch.device = policy_device
    n_envs: int = env.num_envs

    # ------------------------------------------------------------------
    # 1. Per‑env running stats -----------------------------------------
    # ------------------------------------------------------------------
    env_reward = np.zeros(n_envs, dtype=np.float32)
    env_length = np.zeros(n_envs, dtype=np.int32)

    rollout_durations: deque[float] = deque(maxlen=stats_window_size)  # Store last 100 rollout durations
    episode_reward_deque: deque[float] = deque(maxlen=stats_window_size)
    episode_length_deque: deque[int] = deque(maxlen=stats_window_size)
    
    total_rollouts: int = 0
    # First observation ------------------------------------------------------
    obs = env.reset()

    # ------------------------------------------------------------------
    # 2. Helper fns ----------------------------------------------------
    # ------------------------------------------------------------------
    def _bootstrap_value(obs_np: np.ndarray) -> np.ndarray: # TODO: is rlzoo rollout collector infering values during rollouts?
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        return policy_model.v(obs_t).squeeze(-1).cpu().numpy()

    # ------------------------------------------------------------------
    # 3. Main generator loop -------------------------------------------
    # ------------------------------------------------------------------
    total_steps = 0
    total_episodes = 0

    while True:
        # Reset the rollout buffers -----------------------------------
        obs_buf: list[np.ndarray] = []
        actions_buf: list[np.ndarray] = []
        rewards_buf: list[np.ndarray] = []
        values_buf: list[np.ndarray] = []
        done_buf: list[np.ndarray] = []
        logprobs_buf: list[np.ndarray] = []
        frame_buf: list[Sequence[np.ndarray]] = []

        env_step_calls = 0
        rollout_steps = 0
        rollout_episodes = 0

        # --------------------------------------------------------------
        # 3.1 Collect one rollout -------------------------------------
        # --------------------------------------------------------------
        with inference_ctx(policy_model):
            rollout_start = time.time()
            while True:
                # ▸ Policy step — note: **no value call here** ----------------
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action_t, logp_t, value_t = policy_model.act(obs_t)
                action_np = action_t.cpu().numpy()
                action_logp_np = logp_t.cpu().numpy()
                value_np = value_t.cpu().numpy()

                # Environment step ----------------------------------------
                next_obs, reward, done, _ = env.step(action_np)

                # Per‑env episodic bookkeeping ----------------------------
                for i, r in enumerate(reward):
                    env_reward[i] += r
                    env_length[i] += 1
                    if done[i]:
                        _env_reward = float(env_reward[i])
                        _env_length = int(env_length[i])
                        episode_reward_deque.append(_env_reward)
                        episode_length_deque.append(_env_length)
                        env_reward[i] = 0.0
                        env_length[i] = 0

                # TODO: assert values len matches n_envs
                
                # Buffer write -------------------------------------------
                obs_buf.append(obs.copy())  # defensive copy — some envs mutate obs
                actions_buf.append(action_np)
                logprobs_buf.append(action_logp_np)
                rewards_buf.append(reward)
                values_buf.append(value_np)
                done_buf.append(done)
                if collect_frames:
                    frame_buf.append(env.get_images())

                # Advance --------------------------------------------------
                obs = next_obs
                env_step_calls += 1
                rollout_steps += n_envs
                rollout_episodes += done.sum().item()

                # Stop condition ------------------------------------------
                if n_steps is not None and env_step_calls >= n_steps:
                    break
                if n_episodes is not None and rollout_episodes >= n_episodes:
                    break

            total_steps += rollout_steps
            total_episodes += rollout_episodes

            # --------------------------------------------------------------
            # 3.2 End‑of‑rollout processing -------------------------------
            # --------------------------------------------------------------
            # 3.2.1  Stack buffers ----------------------------------------
            obs_arr = np.stack(obs_buf)          # (T, E, obs_dim)
            actions_arr = np.stack(actions_buf)  # (T, E)
            logprobs_arr = np.stack(logprobs_buf)
            rewards_arr = np.stack(rewards_buf)
            values_arr = np.stack(values_buf)
            dones_arr = np.stack(done_buf)
            T = obs_arr.shape[0]

            # Bootstrap value for the *next* state ------------------------
            # TODO: beware of boostrapping for terminal states
            next_value = values_arr[-1]
            # TODO: restore bootstrapping
            #next_value = _bootstrap_value(obs)  # shape: (E,)

            # 3.2.3  GAE‑λ advantages & returns ---------------------------
            advantages_arr = np.zeros_like(rewards_arr, dtype=np.float32)
            gae = np.zeros(n_envs, dtype=np.float32)
            non_terminal = 1.0 - dones_arr.astype(np.float32)
            next_non_terminal = non_terminal[-1]

            for t in reversed(range(T)):
                delta = rewards_arr[t] + gamma * next_value * next_non_terminal - values_arr[t]
                gae = delta + gamma * gae_lambda * gae * next_non_terminal
                advantages_arr[t] = gae
                next_value = values_arr[t]
                next_non_terminal = non_terminal[t]

            returns_arr = advantages_arr + values_arr

            if normalize_advantage:
                adv_flat = advantages_arr.reshape(-1)
                advantages_arr = (advantages_arr - adv_flat.mean()) / (
                    adv_flat.std() + advantages_norm_eps
                )

            # 3.2.4  Env‑major flattening --------------------------------
            obs_env_major = obs_arr.transpose(1, 0, 2)  # (E, T, obs_dim)
            states = torch.as_tensor(
                obs_env_major.reshape(n_envs * T, -1), dtype=torch.float32
            )

            def _flat_env_major(arr: np.ndarray, dtype: torch.dtype):
                return torch.as_tensor(arr.transpose(1, 0).reshape(-1), dtype=dtype)

            actions = _flat_env_major(actions_arr, torch.int64)
            rewards = _flat_env_major(rewards_arr, torch.float32)
            dones = _flat_env_major(dones_arr, torch.bool)
            logps = _flat_env_major(logprobs_arr, torch.float32)
            values = _flat_env_major(values_arr, torch.float32)
            advantages = _flat_env_major(advantages_arr, torch.float32)
            returns = _flat_env_major(returns_arr, torch.float32)

            # 3.2.5  Frames (optional) ------------------------------------
            if collect_frames:
                frames_env_major: list[np.ndarray] = [
                    frame_buf[t][e] for e in range(n_envs) for t in range(T)
                ]
            else:
                frames_env_major = [0] * (n_envs * T)

            # 3.2.6  Yield -------------------------------------------------
            # TODO: yield named tuple
            trajectories = (
                states,
                actions,
                rewards,
                dones,
                logps,
                values,
                advantages,
                returns,
                frames_env_major,
            )

            # Only calculate mean values when window is full
            ep_rew_mean = float(np.mean(episode_reward_deque)) if episode_reward_deque else 0.0
            ep_len_mean = int(np.mean(episode_length_deque)) if episode_length_deque else 0.0
            
            total_rollouts += 1
            rollout_elapsed = time.time() - rollout_start
            rollout_durations.append(rollout_elapsed)
            elapsed_mean = float(np.mean(rollout_durations))
            
            stats = {
                "total_timesteps": total_steps,
                "total_episodes": total_episodes,
                "total_rollouts" : total_rollouts,
                "steps": rollout_steps,
                "episodes": rollout_episodes,
                "elapsed_mean": elapsed_mean,
                "ep_rew_mean": ep_rew_mean,
                "ep_len_mean": ep_len_mean
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
    def __init__(self, env, policy_model, deterministic=False, n_steps=None, n_episodes=None, stats_window_size=100, **kwargs):
        self.env = env
        self.policy_model = policy_model
        self.deterministic = deterministic
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
    
    def _ensure_generator(self, n_episodes=None, n_steps=None, deterministic=None, collect_frames=False):
        if self._generator: return self._generator

        n_episodes = n_episodes if n_episodes is not None else self.n_episodes
        n_steps = n_steps if n_steps is not None else self.n_steps
        deterministic = deterministic if deterministic is not None else self.deterministic
        self._generator = _collect_rollouts( # TODO: don't return tensors, return numpy arrays?
            self.env,
            self.policy_model,
            n_steps=n_steps,
            n_episodes=n_episodes,
            deterministic=deterministic,
            stats_window_size=self.stats_window_size,
            collect_frames=collect_frames,
            **self.kwargs
        )
        return self._generator
    
    def __del__(self):
        if self._generator is None: return 
        self._generator.close() # TODO: drop generator pattern, merge code into class
        self.env.close() # TODO: is this the responsibility of rollout collector or whoever created it?
