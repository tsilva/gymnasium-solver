"""ALE Action Masking Wrapper

Enables full action space (18 actions) for ALE/Atari environments and provides
action masks to prevent the policy from selecting useless actions.

This standardizes action spaces across all Atari games for better generalizability,
similar to how VizDoom uses a standardized action space with button mapping.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ALEActionMaskingWrapper(gym.Wrapper):
    """Wrapper that enables full action space with action masking for ALE environments.

    Uses the full 18-action space available in ALE and provides an action mask
    in the info dict to indicate which actions are meaningful for the current game.

    This allows:
    1. Standardized action spaces across all Atari games
    2. Transfer learning between games
    3. Policy learns to respect action masks (when mask-aware sampling is used)

    The action mask is a binary array of shape (18,) where:
    - 1 = action is meaningful for this game
    - 0 = action has no effect or is useless

    Example:
        For Breakout, only 4 actions are meaningful: [NOOP, FIRE, RIGHT, LEFT]
        The action mask will have 1s at indices [0, 1, 3, 4] and 0s elsewhere.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Verify this is an ALE environment
        assert hasattr(env.unwrapped, 'ale'), "ALEActionMaskingWrapper requires an ALE environment"

        # Get the minimal action set (meaningful actions for this game)
        minimal_actions = env.unwrapped.get_action_meanings()

        # Get full action set (should be 18 for ALE)
        # We'll construct this from the standard ALE full action space
        full_actions = self._get_full_action_meanings()
        assert len(full_actions) == 18, f"Expected 18 full actions, got {len(full_actions)}"

        # Build action mask: 1 if action is in minimal set, 0 otherwise
        self._action_mask = np.zeros(18, dtype=np.int8)
        for action_name in minimal_actions:
            if action_name in full_actions:
                action_idx = full_actions.index(action_name)
                self._action_mask[action_idx] = 1

        # Verify at least one action is valid
        assert self._action_mask.sum() > 0, "No valid actions found in minimal action set"

        # Store for inspection
        self._minimal_actions = minimal_actions
        self._full_actions = full_actions
        self._valid_action_indices = np.where(self._action_mask == 1)[0].tolist()

        # Update action space to use full space (18 actions)
        self.action_space = spaces.Discrete(18)

    def _get_full_action_meanings(self) -> list[str]:
        """Returns the standard ALE full action space meanings.

        All ALE games use the same 18-action full space with this ordering.
        """
        return [
            'NOOP',
            'FIRE',
            'UP',
            'RIGHT',
            'LEFT',
            'DOWN',
            'UPRIGHT',
            'UPLEFT',
            'DOWNRIGHT',
            'DOWNLEFT',
            'UPFIRE',
            'RIGHTFIRE',
            'LEFTFIRE',
            'DOWNFIRE',
            'UPRIGHTFIRE',
            'UPLEFTFIRE',
            'DOWNRIGHTFIRE',
            'DOWNLEFTFIRE',
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Add action mask to info
        info['action_mask'] = self._action_mask.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add action mask to info
        info['action_mask'] = self._action_mask.copy()
        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Returns the action mask for this environment."""
        return self._action_mask.copy()

    def get_valid_actions(self) -> list[int]:
        """Returns list of valid action indices."""
        return self._valid_action_indices.copy()

    def get_action_meanings(self) -> list[str]:
        """Returns action meanings for all 18 actions."""
        return self._full_actions.copy()

    def get_minimal_action_meanings(self) -> list[str]:
        """Returns the minimal (meaningful) action set for this game."""
        return self._minimal_actions.copy()
