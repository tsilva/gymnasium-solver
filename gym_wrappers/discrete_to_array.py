import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteToArray(gym.ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        # TODO: assert discrete observation space

        self.observation_space = spaces.Box(
            low=0,
            high=self.observation_space.n,
            shape=(1,),
            dtype=self.observation_space.dtype
        )
    
    def observation(self, observation):
        return np.array([observation])
