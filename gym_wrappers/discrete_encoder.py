import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteEncoder(gym.ObservationWrapper):
    """
    Unified encoder for discrete observations.

    Supports three encodings:
    - 'array':  shape (1,), integer dtype from Discrete space
    - 'binary': shape (ceil(log2(n)),), float32 0/1 bits
    - 'onehot': shape (n,), float32 one-hot vector

    Args:
        env: Base environment with a Discrete observation space
        encoding: 'array' | 'binary' | 'onehot' (default: 'binary')
    """

    def __init__(self, env, encoding: str = "binary"):
        super().__init__(env)

        # Ensure we have a discrete observation space
        if not isinstance(env.observation_space, spaces.Discrete):
            raise ValueError(
                f"DiscreteEncoder supports only Discrete observation spaces, got {type(env.observation_space)}"
            )

        if encoding not in ("array", "binary", "onehot"):
            raise ValueError("encoding must be one of {'array','binary','onehot'}")

        self.encoding = encoding
        self.n_states = env.observation_space.n

        if encoding == "array":
            # Keep dtype consistent with the original Discrete space
            # Note: upper bound mirrors prior wrapper (high=n) for compatibility.
            self.observation_space = spaces.Box(
                low=0,
                high=self.n_states,  # preserves previous behavior
                shape=(1,),
                dtype=env.observation_space.dtype,  # type: ignore[attr-defined]
            )
        elif encoding == "binary":
            # Minimum number of bits to represent [0, n_states-1]
            self.n_bits = max(1, int(math.ceil(math.log2(self.n_states))))
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_bits,),
                dtype=np.float32,
            )
        else:  # onehot
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_states,),
                dtype=np.float32,
            )

    def observation(self, observation):
        if self.encoding == "array":
            return np.array([observation], dtype=self.observation_space.dtype)
        elif self.encoding == "binary":
            # LSB-first binary representation
            vec = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            for i in range(vec.shape[0]):
                vec[i] = float((observation >> i) & 1)
            return vec
        else:  # onehot
            vec = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            vec[int(observation)] = 1.0
            return vec

