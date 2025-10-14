import gymnasium as gym
import numpy as np
from typing import Union


class FrameSkipWrapper(gym.Wrapper):
    """Skip frames by repeating actions and accumulating rewards.

    Supports both fixed and stochastic (range-based) frame skipping. When a range
    is provided, the number of frames to skip is randomly sampled for each step.

    Args:
        env: Environment to wrap
        skip: Either an int for fixed frame skip, or a tuple (min, max) for
              stochastic frame skip. When using a range, a random value is
              sampled uniformly from [min, max] (inclusive) for each step.

    Usage via registry (YAML):
        # Fixed frame skip (repeat action 4 times)
        - id: FrameSkipWrapper
          skip: 4

        # Stochastic frame skip (randomly skip 2-5 frames each step)
        - id: FrameSkipWrapper
          skip: [2, 5]

    Notes:
    - Actions are repeated for the specified number of frames
    - Rewards are accumulated during skipped frames
    - Episode terminates immediately if terminal/truncated during skip
    - Apply BEFORE frame stacking so frame stack sees effective observations
    - Final observation returned is from the last frame of the skip sequence
    """

    def __init__(self, env: gym.Env, skip: Union[int, tuple[int, int], list[int]]):
        super().__init__(env)

        # Normalize list to tuple for YAML compatibility
        if isinstance(skip, list):
            if len(skip) != 2:
                raise ValueError(f"skip list must have exactly 2 elements, got {len(skip)}")
            skip = tuple(skip)

        # Validate skip parameter
        if isinstance(skip, int):
            if skip < 1:
                raise ValueError(f"skip must be at least 1, got {skip}")
            self._skip_min = skip
            self._skip_max = skip
            self._stochastic = False
        elif isinstance(skip, tuple) and len(skip) == 2:
            skip_min, skip_max = skip
            if not (isinstance(skip_min, (int, np.integer)) and isinstance(skip_max, (int, np.integer))):
                raise TypeError(f"skip range values must be integers, got {type(skip_min)}, {type(skip_max)}")
            if skip_min < 1:
                raise ValueError(f"skip_min must be at least 1, got {skip_min}")
            if skip_max < skip_min:
                raise ValueError(f"skip_max ({skip_max}) must be >= skip_min ({skip_min})")
            self._skip_min = int(skip_min)
            self._skip_max = int(skip_max)
            self._stochastic = True
        else:
            raise TypeError(f"skip must be an int or tuple of 2 ints, got {type(skip)}")

    def step(self, action):
        """Repeat action for skip frames and accumulate rewards."""
        # Sample number of frames to skip
        if self._stochastic:
            n_skip = self.np_random.integers(self._skip_min, self._skip_max + 1)
        else:
            n_skip = self._skip_min

        # Accumulate rewards over skipped frames
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Execute the action for n_skip frames
        for _ in range(n_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Stop immediately if episode ends
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
