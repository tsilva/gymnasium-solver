import os
import gymnasium as gym
from dataclasses import asdict
from utils.io import read_yaml

class EnvInfoWrapper(gym.ObservationWrapper):

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._obs_type = kwargs.get('obs_type', None)

    def get_id(self):
        return self.env.spec.id

    def _get_spec__file(self):
        env_id = self.get_id()
        spec_path = f"config/environments/{env_id}.spec.yaml"
        if not os.path.exists(spec_path): return {}
        try:
            spec = read_yaml(spec_path) or {}
        except Exception:
            spec = {}
        return spec

    def _get_spec__env(self):
        return asdict(self.env.spec)

    def get_spec(self):
        _file_spec = self._get_spec__file()
        _env_spec = self._get_spec__env()
        spec = {**_env_spec, **_file_spec}
        return spec

    def get_obs_type(self):
        return self._obs_type

    def is_rgb_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'rgb'

    def is_ram_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'ram'

    def is_objects_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'objects' # TODO: dict env?

    def get_max_episode_steps(self):
        spec = self.get_spec()
        max_episode_steps = spec['max_episode_steps']
        return max_episode_steps

    def get_reward_treshold(self):
        spec = self.get_spec()
        reward_threshold = spec['reward_threshold']
        return reward_threshold

    # NOTE: required by ObservationWrapper
    def observation(self, observation): 
        return observation

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env_info = EnvInfoWrapper(env)
    print(env_info.get_spec())
    print(env_info.get_reward_treshold())
