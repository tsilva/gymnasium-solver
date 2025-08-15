import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteToBinary(gym.ObservationWrapper):
    """
    Converts discrete observations to binary encoded vectors (one-hot or binary).
    
    This wrapper is useful for environments like Taxi-v3 that have discrete
    observation spaces but where you want to use algorithms that expect
    continuous/vector observations.
    
    Args:
        env: The environment to wrap
        encoding: Either 'binary' or 'onehot'. Defaults to 'binary'.
            - 'binary': Uses minimal bits to represent the discrete value
            - 'onehot': Creates a vector with one 1 and rest 0s
        
    Example:
        >>> env = gym.make('Taxi-v3')  # 500 discrete states
        >>> env_binary = DiscreteBinaryEncoder(env, encoding='binary')
        >>> obs, _ = env_binary.reset()
        >>> print(obs.shape)  # (9,) - ceil(log2(500)) bits
        
        >>> env_onehot = DiscreteBinaryEncoder(env, encoding='onehot')
        >>> obs, _ = env_onehot.reset()
        >>> print(obs.shape)  # (500,) - one-hot vector
    """
    
    def __init__(self, env, encoding='binary'):
        super().__init__(env)
        
        # Ensure we have a discrete observation space
        if not isinstance(env.observation_space, spaces.Discrete):
            raise ValueError(f"DiscreteBinaryEncoder wrapper only supports Discrete observation spaces, "
                           f"got {type(env.observation_space)}")
        
        if encoding not in ['binary', 'onehot']:
            raise ValueError(f"encoding must be 'binary' or 'onehot', got {encoding}")
        
        self.encoding = encoding
        self.n_states = env.observation_space.n
        
        if encoding == 'binary':
            # Calculate minimum number of bits needed
            self.n_bits = max(1, math.ceil(math.log2(self.n_states)))
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_bits,),
                dtype=np.float32
            )
        else:  # onehot
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.n_states,),
                dtype=np.float32
            )
    
    def observation(self, observation):
        """Convert discrete observation to binary or one-hot vector."""
        if self.encoding == 'binary':
            # Convert to binary representation
            binary_repr = np.zeros(self.n_bits, dtype=np.float32)
            for i in range(self.n_bits):
                binary_repr[i] = float((observation >> i) & 1)
            return binary_repr
        else:  # onehot
            # Create one-hot vector
            one_hot = np.zeros(self.n_states, dtype=np.float32)
            one_hot[observation] = 1.0
            return one_hot
