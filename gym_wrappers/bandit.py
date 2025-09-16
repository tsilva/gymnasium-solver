"""Custom Multi-Armed Bandit environment (context-free).

This is a stateless bandit for debugging RL algorithm correctness. Each step
is an arm pull that returns a stochastic reward drawn from a per-arm
distribution. Episodes are short (default length=1) so each step is a full
episode, making value/advantage computation straightforward.

Observation
-----------
- Constant zero vector of shape (n_arms,) each step. This avoids special-cases
  in the policy factory that treat (1,) as an embedding size while keeping the
  policy input deterministic.

Action
------
- Discrete(n_arms): choose which arm to pull.

Reward
------
- Gaussian per-arm by default: N(mean[i], std[i]). You can pass explicit
  `means` (list[float]) and `stds` (float or list[float]) via env_kwargs.

Termination
-----------
- Terminates after `episode_length` steps (default 1). This ensures
  evaluation loops observe episode boundaries without relying on external
  time-limit wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class BanditSpec:
    n_arms: int = 10
    means: Optional[List[float]] = None
    stds: Optional[List[float] | float] = 1.0
    episode_length: int = 1


class MultiArmedBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    @dataclass
    class EnvSpec:
        id: str
        max_episode_steps: int
        reward_threshold: Optional[float] = None

    def __init__(self, *, n_arms: int = 10, means: Optional[List[float]] = None,
                 stds: Optional[List[float] | float] = 1.0, episode_length: int = 1,
                 seed: Optional[int] = None):
        super().__init__()
        if n_arms <= 1:
            raise ValueError("n_arms must be >= 2")
        if episode_length <= 0:
            raise ValueError("episode_length must be >= 1")

        self.spec_cfg = BanditSpec(
            n_arms=int(n_arms),
            means=list(means) if means is not None else None,
            stds=stds,
            episode_length=int(episode_length),
        )

        # RNG
        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.default_rng(self._seed)

        # Action/Observation spaces
        self.action_space = spaces.Discrete(self.spec_cfg.n_arms)
        # Constant zeros; not (1,) to avoid policy_factory's special-case
        self.observation_space = spaces.Box(
            low=0.0, high=0.0, shape=(self.spec_cfg.n_arms,), dtype=np.float32
        )

        # Resolve per-arm parameters
        n = self.spec_cfg.n_arms
        if self.spec_cfg.means is None:
            # Default: linearly spaced around 0 with a clear optimum
            self._means = np.linspace(-0.5, 0.5, num=n, dtype=np.float32)
        else:
            if len(self.spec_cfg.means) != n:
                raise ValueError("means must be length n_arms")
            self._means = np.asarray(self.spec_cfg.means, dtype=np.float32)

        if isinstance(self.spec_cfg.stds, (int, float)):
            self._stds = np.full(n, float(self.spec_cfg.stds), dtype=np.float32)
        else:
            _stds = list(self.spec_cfg.stds)
            if len(_stds) != n:
                raise ValueError("stds list must be length n_arms")
            self._stds = np.asarray(_stds, dtype=np.float32)

        self._t = 0

        # Minimal EnvSpec to cooperate with EnvInfoWrapper helpers
        self.spec = MultiArmedBanditEnv.EnvSpec(
            id="Bandit-v0",
            max_episode_steps=self.spec_cfg.episode_length,
            reward_threshold=None,
        )

    # Gymnasium API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        self._t = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "n_arms": self.spec_cfg.n_arms,
            "means": self._means.copy(),
            "stds": self._stds.copy(),
        }
        return obs, info
    
    def render(self):
        import torch
        from PIL import Image

        # Create tensor in CHW format
        tensor = torch.zeros((3, 84, 84), dtype=torch.uint8)

        # Convert to HWC format for PIL
        array = tensor.permute(1, 2, 0).numpy()

        # Create PIL image
        img = Image.fromarray(array)

        return img
    def step(self, action: int):
        if not self.action_space.contains(action):
            raise AssertionError("Action out of bounds")
        # Sample reward from the arm's distribution
        mean = float(self._means[int(action)])
        std = float(self._stds[int(action)])
        reward = float(self._rng.normal(loc=mean, scale=std))

        self._t += 1
        terminated = self._t >= self.spec_cfg.episode_length
        truncated = False

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "chosen_arm": int(action),
            "optimal_arm": int(np.argmax(self._means)),
            "optimal_mean": float(np.max(self._means)),
        }
        return obs, reward, terminated, truncated, info

    # Optional: support Gymnasium's seed() legacy
    def seed(self, seed: Optional[int] = None):  # pragma: no cover - compatibility
        self._seed = int(seed) if seed is not None else self._seed
        self._rng = np.random.default_rng(self._seed)
        return [self._seed]
