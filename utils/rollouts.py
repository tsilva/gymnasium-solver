import itertools
from collections import deque
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple

import time
import numpy as np
import torch
from torch.distributions import Categorical

# TODO: move this somewhere else?
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

# TODO: move this somewhere else?
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
def _collect_rollouts(
    env,
    policy_model: torch.nn.Module,
    *,
    value_model: Optional[torch.nn.Module] = None,
    n_steps: Optional[int] = None,
    n_episodes: Optional[int] = None,
    deterministic: bool = False,
    gamma: float = 0.99, # TODO: make sure these are passed correctly
    gae_lambda: float = 0.95, # TODO: make sure these are passed correctly
    normalize_advantage: bool = True,
    advantages_norm_eps: float = 1e-8,
    collect_frames: bool = False,
    mean_reward_window: int = 100, # TODO: make sure these are passed correctly
    mean_length_window: int = 100# TODO: make sure these are passed correctly
):
    # ------------------------------------------------------------------
    # 0. Sanity checks --------------------------------------------------
    # ------------------------------------------------------------------
    assert (n_steps is not None and n_steps > 0) or (
        n_episodes is not None and n_episodes > 0
    ), "Provide *n_steps*, *n_episodes*, or both (> 0)."

    policy_device = _device_of(policy_model)
    #assert policy_device.type != 'cpu', "Policy model must be on GPU or MPS, not CPU."
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

    rollout_durations: deque[float] = deque(maxlen=100)  # Store last 100 rollout durations
    episode_reward_deque: deque[float] = deque(maxlen=mean_reward_window)
    episode_length_deque: deque[int] = deque(maxlen=mean_length_window)
    
    total_rollouts: int = 0
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
    total_steps = 0
    total_episodes = 0

    while True:
        # Reset the rollout buffers -----------------------------------
        obs_buf: list[np.ndarray] = []
        actions_buf: list[np.ndarray] = []
        rewards_buf: list[np.ndarray] = []
        done_buf: list[np.ndarray] = []
        logprobs_buf: list[np.ndarray] = []
        frame_buf: list[Sequence[np.ndarray]] = []

        env_step_calls = 0
        rollout_steps = 0
        rollout_episodes = 0

        # --------------------------------------------------------------
        # 3.1 Collect one rollout -------------------------------------
        # --------------------------------------------------------------
        with inference_ctx(policy_model, value_model):
            rollout_start = time.time()
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
            mean_ep_reward = float(np.mean(episode_reward_deque)) if episode_reward_deque else 0.0
            mean_ep_length = float(np.mean(episode_length_deque)) if episode_length_deque else 0.0
            
            total_rollouts += 1
            rollout_elapsed = time.time() - rollout_start
            rollout_durations.append(rollout_elapsed)
            rollout_mean_duration = np.mean(rollout_durations)
            
            stats = {
                "total_rollouts" : total_rollouts,
                "total_episodes": total_episodes,
                "total_timesteps": total_steps,
                "rollout_step_count": rollout_steps,
                "rollout_episode_count": rollout_episodes,
                "rollout_mean_duration": rollout_mean_duration,
                "mean_ep_reward": mean_ep_reward,
                "mean_ep_length": mean_ep_length
            }

        yield trajectories, stats

class RolloutCollector():
    # TODO: how do they perform eval, at which cadence?
    def __init__(self, _id, env, policy_model, value_model=None, deterministic=False, n_steps=None, n_episodes=None, **kwargs):
        self._id = _id
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.deterministic = deterministic
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self._generator = None
        self.stats = {}
        self.kwargs = kwargs

    def collect(self, *args, **kwargs):
        generator = self._ensure_generator(*args, **kwargs)
        trajectories, stats = next(generator)
        self.stats = stats
        return trajectories
    
    def get_stats(self):
        return self.stats
    
    def get_reward_threshold(self):
        from utils.environment import get_env_reward_threshold
        reward_threshold = get_env_reward_threshold(self.env)
        return reward_threshold
    
    def get_mean_ep_reward(self):
        return self.stats.get('mean_ep_reward', 0.0)
    
    def get_total_episodes(self):
        return self.stats.get('total_episodes', 0)
    
    def get_mean_reward_window(self):
        return 100 # TODO: softcode
    
    def is_reward_threshold_reached(self):
        reward_threshold = self.get_reward_threshold()
        mean_ep_reward = self.get_mean_ep_reward()
        mean_reward_window = self.get_mean_reward_window()
        total_episodes = self.get_total_episodes()
        reached = total_episodes >= mean_reward_window and mean_ep_reward >= reward_threshold
        return reached
    
    def _ensure_generator(self, n_episodes=None, n_steps=None, deterministic=None, collect_frames=False):
        if self._generator: return self._generator

        n_episodes = n_episodes if n_episodes is not None else self.n_episodes
        n_steps = n_steps if n_steps is not None else self.n_steps
        deterministic = deterministic if deterministic is not None else self.deterministic
        self._generator = _collect_rollouts( # TODO: don't return tensors, return numpy arrays?
            self.env,
            self.policy_model,
            value_model=self.value_model,
            n_steps=n_steps,
            n_episodes=n_episodes,
            deterministic=deterministic,
            collect_frames=collect_frames,
            **self.kwargs
        )
        return self._generator
    
    def __del__(self):
        if self._generator is None: return 
        self._generator.close()
        self.env.close()

    # TODO: add env spec details
    def __str__(self) -> str:
        """Return a human-readable string representation of the rollout collector."""
        lines = [f"RolloutCollector '{self._id}'", "=" * (len(f"RolloutCollector '{self._id}'")), ""]
        
        # Configuration section
        lines.extend([
            "CONFIGURATION:",
            f"  Environment: {getattr(self.env, 'spec', 'Unknown')} ({self.env.num_envs} parallel envs)",
            f"  Policy Model: {self.policy_model.__class__.__name__}",
            f"  Value Model: {self.value_model.__class__.__name__ if self.value_model else 'None'}",
            f"  Mode: {'Deterministic' if self.deterministic else 'Stochastic'}",
            f"  Collection: {self.n_steps or 'None'} steps, {self.n_episodes or 'None'} episodes",
            f"  Generator Active: {'Yes' if self._generator else 'No'}",
            ""
        ])
        
        # Statistics section
        if self.stats:
            lines.extend([
                "CURRENT STATISTICS:",
                f"  Total Rollouts: {self.stats.get('n_rollouts', 0)}",
                f"  Total Episodes: {self.stats.get('n_episodes', 0)}",
                f"  Total Steps: {self.stats.get('n_steps', 0)}",
                f"  Mean Episode Reward: {self.stats.get('mean_ep_reward', 0.0):.3f}",
                f"  Mean Episode Length: {self.stats.get('mean_ep_length', 0.0):.1f}",
                f"  Mean Rollout Duration: {self.stats.get('mean_rollout_duration', 0.0):.3f}s",
                ""
            ])
            
            # Performance metrics
            steps_per_second = self.stats.get('n_steps', 0) / max(self.stats.get('mean_rollout_duration', 1e-6), 1e-6)
            episodes_per_rollout = self.stats.get('n_episodes', 0) / max(self.stats.get('n_rollouts', 1), 1)
            
            lines.extend([
                "PERFORMANCE METRICS:",
                f"  Steps per Second: {steps_per_second:.1f}",
                f"  Episodes per Rollout: {episodes_per_rollout:.1f}",
            ])
        else:
            lines.extend([
                "CURRENT STATISTICS:",
                "  No rollouts collected yet",
            ])
        
        return "\n".join(lines)
