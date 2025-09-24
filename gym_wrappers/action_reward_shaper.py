import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ActionRewardShaper(gym.Wrapper):
    """Add a fixed bonus/penalty to the reward based on the chosen action.

    Usage via registry (YAML):
      - id: ActionRewardShaper
        rewards: [<a0_bonus>, <a1_bonus>, ...]

    Notes:
    - Supports only Discrete action spaces.
    - The length of `rewards` must match the current (possibly wrapped) action space size.
    - The wrapper adds the per-action bonus to the environment's reward and exposes
      it in `info['action_reward_bonus']` for debugging/analysis.
    """

    def __init__(self, env: gym.Env, rewards: list[float]):
        super().__init__(env)
        self._assert_valid_env(env)

        if not isinstance(rewards, (list, tuple)):
            raise TypeError("ActionRewardShaper: 'rewards' must be a list or tuple of floats")

        n_actions = int(env.action_space.n)
        if len(rewards) != n_actions:
            raise ValueError(
                f"ActionRewardShaper: length of rewards ({len(rewards)}) must match action space size ({n_actions})"
            )

        # Store as immutable tuple for safety
        self._rewards = tuple(float(r) for r in rewards)

    def step(self, action):
        # Coerce to Python int (handles numpy scalars/arrays)
        action_idx = int(np.asarray(action).item())

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add action-specific reward bonus
        bonus = self._rewards[action_idx]
        shaped_reward = reward + bonus

        if info is None:
            info = {}
        # Expose bonus for transparency/debugging
        info["action_reward_bonus"] = bonus

        return obs, shaped_reward, terminated, truncated, info

    def _assert_valid_env(self, env: gym.Env):
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(
                f"ActionRewardShaper requires a Discrete action space, got {type(env.action_space)}"
            )

