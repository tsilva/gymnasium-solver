import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StickyActionsWrapper(gym.ActionWrapper):
    """Make specific buttons in MultiBinary action space sticky by forcing them with probability.

    For each specified button index, overrides it to 1 with the given probability,
    regardless of the agent's action. This encourages persistent button presses.

    Usage via registry (YAML):
      - id: StickyActionsWrapper
        sticky_buttons: {7: 0.75, 8: 0.25}  # button_index: probability

    Example:
      sticky_buttons: {7: 0.75} forces button 7 to be pressed 75% of the time

    Notes:
    - Only supports MultiBinary action spaces
    - Probabilities must be in [0.0, 1.0]
    - Original agent actions are modified in-place
    """

    def __init__(self, env: gym.Env, sticky_buttons: dict[int, float]):
        super().__init__(env)
        self._assert_valid_env(env)

        # Validate type before normalization (fail-fast)
        if not isinstance(sticky_buttons, dict):
            raise TypeError("sticky_buttons must be a dict mapping button_index to probability")

        # Normalize string keys to integers (YAML parser converts numeric keys to strings)
        normalized_buttons = {}
        for key, value in sticky_buttons.items():
            if isinstance(key, str) and key.isdigit():
                normalized_buttons[int(key)] = value
            else:
                normalized_buttons[key] = value

        self._assert_valid_sticky_buttons(env, normalized_buttons)

        # Store sticky button config
        self._sticky_buttons = normalized_buttons
        self._n_buttons = int(env.action_space.n)

    def action(self, action):
        # Convert to mutable numpy array if needed
        action = np.asarray(action, dtype=self.action_space.dtype)
        if action.shape != (self._n_buttons,):
            raise ValueError(f"Action shape {action.shape} doesn't match expected ({self._n_buttons},)")

        # Make sticky buttons active with specified probability
        for button_idx, prob in self._sticky_buttons.items():
            if self.np_random.random() < prob:
                action[button_idx] = 1

        return action

    def _assert_valid_env(self, env: gym.Env):
        if not isinstance(env.action_space, spaces.MultiBinary):
            raise TypeError(
                f"StickyActionsWrapper requires MultiBinary action space, got {type(env.action_space)}"
            )

    def _assert_valid_sticky_buttons(self, env: gym.Env, sticky_buttons: dict[int, float]):
        if len(sticky_buttons) == 0:
            raise ValueError("sticky_buttons must contain at least one entry")

        n_buttons = int(env.action_space.n)

        for button_idx, prob in sticky_buttons.items():
            if not isinstance(button_idx, (int, np.integer)):
                raise TypeError(f"Button index {button_idx} must be an integer")

            if button_idx < 0 or button_idx >= n_buttons:
                raise ValueError(
                    f"Button index {button_idx} out of range [0, {n_buttons - 1}]"
                )

            if not isinstance(prob, (int, float, np.number)):
                raise TypeError(f"Probability for button {button_idx} must be numeric")

            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"Probability {prob} for button {button_idx} must be in [0.0, 1.0]"
                )
