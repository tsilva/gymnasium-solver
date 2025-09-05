import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnvInfoWrapper(gym.ObservationWrapper):
   
   def get_id(self):
      spec = self.env.get_spec()
      print(spec)
      return env.id
   
   def get_spec(self):
      env_id = self.get_id()
      spec = gym.spec(env_id)
      return spec

   def get_reward_treshold(self):
       return self.env.get_reward_threshold()

env = gym.make("CartPole-v1")
env_info = EnvInfoWrapper(env)
print(env_info.get_spec())
print(env_info.get_reward_treshold())
