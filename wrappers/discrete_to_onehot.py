import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteToOneHot(gym.ObservationWrapper):
    """
    Converts discrete observations to one-hot encoded vectors.
    
    This wrapper is useful for environments like Taxi-v3 that have discrete
    observation spaces but where you want to use algorithms that expect
    continuous/vector observations.
    
    Args:
        env: The environment to wrap
        
    Example:
        >>> env = gym.make('Taxi-v3')
        >>> env = DiscreteToOneHot(env)
        >>> obs, _ = env.reset()
        >>> print(obs.shape)  # (500,) instead of scalar
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Ensure we have a discrete observation space
        if not isinstance(env.observation_space, spaces.Discrete):
            raise ValueError(f"DiscreteToOneHot wrapper only supports Discrete observation spaces, "
                           f"got {type(env.observation_space)}")
        
        # Store the number of discrete states
        self.n_states = env.observation_space.n
        
        # Create new observation space as a Box with one-hot vectors
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_states,),
            dtype=np.float32
        )
    
    def observation(self, observation):
        """Convert discrete observation to one-hot vector."""
        # Create one-hot vector
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[observation] = 1.0
        return one_hot


# Alias for convenience
OneHotWrapper = DiscreteToOneHot
