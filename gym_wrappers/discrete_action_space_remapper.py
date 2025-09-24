import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteActionSpaceRemapperWrapper(gym.ActionWrapper):
    """Remap a Discrete action space to a smaller Discrete via index mapping."""

    def __init__(self, env: gym.Env, mapping: list[int]):
        super().__init__(env)
        
        # Assert that the environment and mapping are valid
        self._assert_valid_env(env)
        self._assert_valid_mapping(env, mapping)

        # Store an immutable copy of the mapping
        self._mapping = tuple(mapping)

        # Adjust the action space to the new mapping
        self._original_action_space = env.action_space
        self.action_space = spaces.Discrete(len(self._mapping))

    def action(self, action):
        # Accept scalar or array-like and coerce to Python int
        action = int(np.asarray(action).item())
        if action < 0 or action >= len(self._mapping): raise ValueError(f"Action index {action} is out of range for remapped space with n={len(self._mapping)}")
        return self._mapping[action]

    def _assert_valid_env(self, env: gym.Env):
        # Assert that wrapper is being applied to an action space that is Discrete
        if not isinstance(env.action_space, spaces.Discrete): raise TypeError(f"requires a Discrete action space, got {type(env.action_space)}")
    
    def _assert_valid_mapping(self, env: gym.Env, mapping: list[int]):
        # Assert that mapping is a non-empty list of integers
        if not isinstance(mapping, (list, tuple)) or len(mapping) == 0: raise ValueError("mapping must be a non-empty list of integers")
        
        # Assert that all mapping entries are within the original action space
        n_actions_original = int(env.action_space.n)
        for action in mapping:
            if not isinstance(action, (int, np.integer)): raise TypeError("mapping must contain only integers")
            if action < 0 or action >= n_actions_original: raise ValueError(f"mapping contains invalid action index {action}; valid range is [0, {n_actions_original - 1}]")

