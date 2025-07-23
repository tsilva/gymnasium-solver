import torch
import numpy as np
from typing import Optional, Tuple, Sequence
from torch.utils.data import Dataset as TorchDataset
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from collections import deque
from typing import Optional, Sequence, Tuple
from collections import deque

import numpy as np
import torch
from torch.distributions import Categorical


import torch.nn as nn
from contextlib import contextmanager

from contextlib import contextmanager
import torch.nn as nn
from typing import Optional, Union, List

from contextlib import contextmanager
import torch, itertools

@contextmanager
def inference_ctx(*modules):
    """
    Temporarily puts all passed nn.Module objects in eval mode and
    disables grad-tracking. Restores their original training flag
    afterwards.

    Usage:
        with inference_ctx(actor, critic):
            ... collect trajectories ...
    """
    # Filter out Nones and flatten (in case you pass lists/tuples)
    flat = [m for m in itertools.chain.from_iterable(
            (m if isinstance(m, (list, tuple)) else (m,)) for m in modules)
            if m is not None]

    # Remember original .training flags
    was_training = [m.training for m in flat]
    try:
        for m in flat:
            m.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, flag in zip(flat, was_training):
            if flag:   # only restore if it *was* in train mode
                m.train()

# TODO: is this needed?
def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


# TODO: close envs in the end
# TODO: make rollout collector use its own buffer
# TODO: add # env.normalize_obs() support
# TODO: add # env.normalize_rewards() support
# TODO: batch value inference post rollout
# TODO: consider returning transition object
# TODO: log more stats
# TODO: rename var names
# TODO: look for bugs
# TODO: empirically torch.inference_mode() is not faster than torch.no_grad() for this use case, retest for other envs
@torch.no_grad()
def collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    *,
    value_model: Optional[torch.nn.Module] = None,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    deterministic: bool = False,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize_advantage: bool = True,
    advantages_norm_eps: float = 1e-8,
    collect_frames: bool = False,
):
    # ------------------------------------------------------------------
    # 0. Sanity checks --------------------------------------------------
    # ------------------------------------------------------------------
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    policy_device = _device_of(policy_model)
    assert policy_device.type != 'cpu', "Policy model must be on GPU or MPS, not CPU."
    if value_model is not None:
        value_device = _device_of(value_model)
        assert (
            policy_device == value_device
        ), "Policy and value models must be on the same device."

    device: torch.device = policy_device
    n_envs: int = env.num_envs

    # ------------------------------------------------------------------
    # 1. Per‑env running stats -----------------------------------------
    # ------------------------------------------------------------------
    env_reward = np.zeros(n_envs, dtype=np.float32)
    env_length = np.zeros(n_envs, dtype=np.int32)

    episode_reward_deque: deque[float] = deque(maxlen=100)
    episode_length_deque: deque[int] = deque(maxlen=100)

    # First observation ------------------------------------------------------
    obs = env.reset()

    # ------------------------------------------------------------------
    # 2. Helper fns ----------------------------------------------------
    # ------------------------------------------------------------------
    def _bootstrap_value(obs_np: np.ndarray) -> np.ndarray:
        if value_model is None:
            return np.zeros((n_envs,), dtype=np.float32)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        return value_model(obs_t).squeeze(-1).cpu().numpy()

    def _infer_policy(obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        logits = policy_model(obs_t)
        dist = Categorical(logits=logits)
        action_t = logits.argmax(-1) if deterministic else dist.sample()
        logp_t = dist.log_prob(action_t)
        return action_t.cpu().numpy(), logp_t.cpu().numpy()

    # ------------------------------------------------------------------
    # 3. Main generator loop -------------------------------------------
    # ------------------------------------------------------------------
    total_step_count = 0
    total_episode_count = 0

    while True:
        # Reset the rollout buffers -----------------------------------
        obs_buf: list[np.ndarray] = []
        actions_buf: list[np.ndarray] = []
        rewards_buf: list[np.ndarray] = []
        done_buf: list[np.ndarray] = []
        logprobs_buf: list[np.ndarray] = []
        frame_buf: list[Sequence[np.ndarray]] = []

        rollout_step_count = 0
        rollout_episode_count = 0

        # --------------------------------------------------------------
        # 3.1 Collect one rollout -------------------------------------
        # --------------------------------------------------------------
        with inference_ctx(policy_model, value_model):
            while True:
                # ▸ Policy step — note: **no value call here** ----------------
                action_np, action_logp_np = _infer_policy(obs)

                # Environment step ----------------------------------------
                next_obs, reward, done, _ = env.step(action_np)

                # Per‑env episodic bookkeeping ----------------------------
                for i, r in enumerate(reward):
                    env_reward[i] += r
                    env_length[i] += 1
                    if done[i]:
                        episode_reward_deque.append(float(env_reward[i]))
                        episode_length_deque.append(int(env_length[i]))
                        env_reward[i] = 0.0
                        env_length[i] = 0

                # Buffer write -------------------------------------------
                obs_buf.append(obs.copy())  # defensive copy — some envs mutate obs
                actions_buf.append(action_np)
                logprobs_buf.append(action_logp_np)
                rewards_buf.append(reward)
                done_buf.append(done)
                if collect_frames:
                    frame_buf.append(env.get_images())

                # Advance --------------------------------------------------
                obs = next_obs
                rollout_step_count += n_envs
                rollout_episode_count += done.sum().item()

                # Stop condition ------------------------------------------
                if n_steps is not None and rollout_step_count >= n_steps:
                    break
                if n_episodes is not None and rollout_episode_count >= n_episodes:
                    break

            total_step_count += rollout_step_count
            total_episode_count += rollout_episode_count

            # --------------------------------------------------------------
            # 3.2 End‑of‑rollout processing -------------------------------
            # --------------------------------------------------------------
            # 3.2.1  Stack buffers ----------------------------------------
            obs_arr = np.stack(obs_buf)          # (T, E, obs_dim)
            actions_arr = np.stack(actions_buf)  # (T, E)
            logprobs_arr = np.stack(logprobs_buf)
            rewards_arr = np.stack(rewards_buf)
            dones_arr = np.stack(done_buf)
            T = obs_arr.shape[0]

            # 3.2.2  **Batched value inference** --------------------------
            if value_model is None:
                values_arr = np.zeros((T, n_envs), dtype=np.float32)
            else:
                obs_flat_t = torch.as_tensor(
                    obs_arr.reshape(T * n_envs, -1),
                    dtype=torch.float32,
                    device=device,
                )
                values_flat = value_model(obs_flat_t).squeeze(-1).cpu().numpy()
                values_arr = values_flat.reshape(T, n_envs)

            # Bootstrap value for the *next* state ------------------------
            next_value = _bootstrap_value(obs)  # shape: (E,)

            # 3.2.3  GAE‑λ advantages & returns ---------------------------
            advantages_arr = np.zeros_like(rewards_arr, dtype=np.float32)
            gae = np.zeros(n_envs, dtype=np.float32)
            non_terminal = 1.0 - dones_arr.astype(np.float32)
            next_non_terminal = non_terminal[-1]

            for t in reversed(range(T)):
                delta = rewards_arr[t] + gamma * next_value * next_non_terminal - values_arr[t]
                gae = delta + gamma * lam * gae * next_non_terminal
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
            stats = {
                "n_episodes": total_episode_count,
                "n_steps": total_step_count,
                "mean_ep_reward": float(np.mean(episode_reward_deque))
                if episode_reward_deque
                else 0.0,
                "mean_ep_length": float(np.mean(episode_length_deque))
                if episode_length_deque
                else 0.0,
            }

        yield trajectories, stats

class SyncRolloutCollector():
    def __init__(self, env, policy_model, value_model=None, deterministic=False, n_steps=None, n_episodes=None):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.deterministic = deterministic
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.generator = None

    # TODO: is this dataset/dataloader update strategy correct?
    def collect(self, n_episodes=None, n_steps=None, deterministic=None, collect_frames=False):
        if not self.generator:
            n_episodes = n_episodes if n_episodes is not None else self.n_episodes
            n_steps = n_steps if n_steps is not None else self.n_steps
            deterministic = deterministic if deterministic is not None else self.deterministic
            self.generator = collect_rollouts( # TODO: don't return tensors, return numpy arrays?
                self.env,
                self.policy_model,
                value_model=self.value_model,
                n_steps=n_steps,
                n_episodes=n_episodes,
                deterministic=deterministic,
                collect_frames=collect_frames
                #last_obs=self.last_obs, # TODO: should I use this?
                #collect_frames=True
            )
        trajectories, stats = next(self.generator)
        #self.dataset.update(*trajectories)
        return trajectories, stats
