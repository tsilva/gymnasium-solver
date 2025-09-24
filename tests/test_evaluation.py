import math
from typing import List

import numpy as np
import torch


class _FakeVecEnv:
    """
    Minimal vectorized env stub for testing evaluate_policy.

    - Synchronous stepping across n_envs
    - Each sub-env produces episodes with predefined lengths (cycles if needed)
    - Rewards are 1.0 per step, so episode return == episode length
    - Emits Gym/Monitor-like info["episode"] on dones: {"r": total_reward, "l": length}
    """

    def __init__(self, per_env_episode_lengths: List[List[int]], obs_dim: int = 3):
        self.num_envs = len(per_env_episode_lengths)
        self._episodes = [list(lengths) for lengths in per_env_episode_lengths]
        assert self.num_envs > 0

        # Pointers/state per env
        self._ep_idx = [0 for _ in range(self.num_envs)]
        self._step_in_ep = [0 for _ in range(self.num_envs)]
        self._obs_dim = obs_dim

    def _current_len(self, i: int) -> int:
        seq = self._episodes[i]
        idx = self._ep_idx[i] % len(seq)
        return int(seq[idx])

    def reset(self):
        self._ep_idx = [0 for _ in range(self.num_envs)]
        self._step_in_ep = [0 for _ in range(self.num_envs)]
        return np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)

    def step(self, actions):  # actions unused
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)
        rewards = np.ones((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=bool)
        infos: List[dict] = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            self._step_in_ep[i] += 1
            ep_len = self._current_len(i)
            if self._step_in_ep[i] >= ep_len:
                # emit episode info and reset sub-episode state
                dones[i] = True
                infos[i] = {"episode": {"r": float(ep_len), "l": int(ep_len)}}
                self._ep_idx[i] += 1
                self._step_in_ep[i] = 0

        return obs, rewards, dones, infos


class _DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Register a parameter so _device_of can locate a device
        self._p = torch.nn.Parameter(torch.zeros(1))
    
    def forward(self, obs_t: torch.Tensor):
        # Always return a distribution that prefers action 0
        logits = torch.zeros((obs_t.shape[0], 2), dtype=torch.float32, device=obs_t.device)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, None


def test_evaluate_policy_single_env_exact_counts():
    from utils.evaluation import evaluate_policy

    # One env with three episodes of lengths 2, 3, 1
    env = _FakeVecEnv([[2, 3, 1]])
    policy = _DummyPolicy()

    metrics = evaluate_policy(env, policy, n_episodes=3, deterministic=True)

    # Expectations
    assert metrics["total_episodes"] == 3
    # total_timesteps is the sum of episode lengths for a single env
    assert metrics["total_timesteps"] == 2 + 3 + 1
    # Means match the episode returns/lengths (reward == length)
    assert math.isclose(metrics["ep_rew/mean"], (2 + 3 + 1) / 3.0)
    assert math.isclose(metrics["ep_len/mean"], (2 + 3 + 1) / 3.0)
    # Per-env diagnostics
    assert metrics["per_env/episodes_0"] == 3
    assert math.isclose(metrics["per_env/ep_rew_mean_0"], (2 + 3 + 1) / 3.0)
    assert math.isclose(metrics["per_env/ep_len_mean_0"], (2 + 3 + 1) / 3.0)
    # Determinism is handled by evaluation via distribution mode; no policy flag assertion


def test_evaluate_policy_balanced_multi_env():
    from utils.evaluation import evaluate_policy

    # Three envs; each episode length is always 2 steps (reward 2)
    env = _FakeVecEnv([[2], [2], [2]])
    policy = _DummyPolicy()

    # Request 10 episodes -> targets should be [4, 3, 3]
    metrics = evaluate_policy(env, policy, n_episodes=10, deterministic=False)

    # Check overall counts and means
    assert metrics["total_episodes"] == 10
    assert math.isclose(metrics["ep_rew/mean"], 2.0)
    assert math.isclose(metrics["ep_len/mean"], 2.0)

    # With episode length 2 and targets [4,3,3], vector steps = 8 => timesteps = 8 * 3
    assert metrics["total_timesteps"] == 8 * 3

    # Per-env episode counts follow balancing logic
    assert metrics["per_env/episodes_0"] == 4
    assert metrics["per_env/episodes_1"] == 3
    assert metrics["per_env/episodes_2"] == 3

    # Per-env means are consistent
    for i in range(3):
        assert math.isclose(metrics[f"per_env/ep_rew_mean_{i}"], 2.0)
        assert math.isclose(metrics[f"per_env/ep_len_mean_{i}"], 2.0)
