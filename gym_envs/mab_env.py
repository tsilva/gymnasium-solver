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
from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class BanditSpec:
    n_arms: int = 10
    means: Optional[List[float]] = None
    stds: Optional[List[float] | float] = 1.0
    episode_length: int = 1


class MultiArmedBanditEnv(gym.Env):
    metadata = {"render_modes": []}

    # TODO: should I reuse gym class?
    @dataclass
    class EnvSpec:
        id: str
        max_episode_steps: int
        reward_threshold: Optional[float] = None

    def __init__(
        self, 
        *, 
        n_arms: int = 10, 
        means: Optional[List[float]] = None,
        stds: Optional[List[float] | float] = 1.0, 
        episode_length: int = 1,
        seed: Optional[int] = None
    ):
        super().__init__()

        # Assert parameters are valid
        if n_arms <= 1: raise ValueError("n_arms must be >= 2")
        if episode_length <= 0: raise ValueError("episode_length must be >= 1")

        # Create spec configuration
        self.spec_cfg = BanditSpec(
            n_arms=int(n_arms),
            means=list(means) if means is not None else None,
            stds=stds,
            episode_length=episode_length
        )

        # Initiialize RNG using provided seed (if any)
        self._init_rng(seed)

        # Create action space (eg: how many slot machine arms can be pulled)
        self.action_space = spaces.Discrete(self.spec_cfg.n_arms)

        # Create observation space (always the same state because this 
        # is a multi-armed bandit, so its a stateless environment)
        # TODO: why the obs shape        
        self.observation_space = spaces.Box(
            low=0.0, high=0.0, shape=(self.spec_cfg.n_arms,), dtype=np.float32
        )

        # Initialize arm distributions
        self._init_means(self.spec_cfg.n_arms, self.spec_cfg.means)
        self._init_stds(self.spec_cfg.n_arms, self.spec_cfg.stds)

        # Set timestep to 0
        self._timestep = 0

        # Minimal EnvSpec to cooperate with EnvInfoWrapper helpers
        reward_threshold = max(self._means)
        self.spec = MultiArmedBanditEnv.EnvSpec(
            id="Bandit-v0",
            max_episode_steps=self.spec_cfg.episode_length,
            reward_threshold=reward_threshold,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Re-initialize RNG if new seed is provided
        if seed is not None: self._init_rng(seed)
        
        # Set timestep to 0
        self._timestep = 0

        # Define dummy observation (stateless env)
        obs = self._build_dummy_observation()
        
        # Define extra info
        info = {
            "n_arms": self.spec_cfg.n_arms,
            "means": self._means.copy(),
            "stds": self._stds.copy(),
        }

        # Return environment reset data
        return obs, info
    
    # TODO: add support for not returning images
    def render(self):
        # Create dummy images just to avoid errors (codebase currently 
        # doesn't support environments that don't support rendering)
        import torch
        from PIL import Image
        tensor = torch.zeros((3, 84, 84), dtype=torch.uint8)
        array = tensor.permute(1, 2, 0).numpy()
        img = Image.fromarray(array)
        return img

    def step(self, action: int):
        # Assert that action is valid
        if not self.action_space.contains(action): raise AssertionError("Action out of bounds")

        # Sample reward from the arm's distribution
        reward = self._sample_reward(action)

        # Advance timestep and check if episode is terminated
        self._timestep += 1
        terminated = self._timestep >= self.spec_cfg.episode_length
        truncated = False

        # Define dummy observation (stateless env)
        obs = self._build_dummy_observation()

        # Define extra info
        info = {
            "chosen_arm": int(action),
            "optimal_arm": int(np.argmax(self._means)),
            "optimal_mean": float(np.max(self._means)),
        }

        # Return environment step data
        return obs, reward, terminated, truncated, info

    def _sample_reward(self, action: int) -> float:
        mean = float(self._means[int(action)])
        std = float(self._stds[int(action)])
        return float(self._rng.normal(loc=mean, scale=std))

    def _build_dummy_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _init_rng(self, seed: Optional[int]):
        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.default_rng(self._seed) if self._seed is not None else np.random.default_rng()
       
    def _init_means(self, n_arms: int, means: Optional[List[float]]):
        if means is None: means = np.linspace(0, n_arms, num=n_arms, dtype=np.float32)
        if len(means) != n_arms: raise ValueError("means must be length n_arms")
        self._means = np.asarray(means, dtype=np.float32)

    def _init_stds(self, n_arms: int, stds: Optional[List[float] | float]):
        if isinstance(stds, (int, float)): stds = np.full(n_arms, float(stds), dtype=np.float32)
        if len(stds) != n_arms: raise ValueError("stds list must be length n_arms")
        self._stds = np.asarray(stds, dtype=np.float32)



        
