import os
from dataclasses import asdict

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

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
        if not os.path.exists(spec_path):
            return {}
        spec = read_yaml(spec_path) or {}
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


    def get_reward_treshold(self):
        spec = self.get_spec()
        reward_threshold = spec['reward_threshold']
        return reward_threshold

    def get_time_limit(self):
        # If the time limit wrapper is found, return the max episode steps
        wrapper = self._find_wrapper(TimeLimit)
        if wrapper: 
            value = getattr(wrapper, "max_episode_steps", None)
            if not value: value = getattr(wrapper, "_max_episode_steps", None)
            if value: return value

        # Otherwise, return the max episode steps from the spec (if available)
        spec = self.get_spec()
        value = spec.get("max_episode_steps", None)
        return value
    
    # TODO: clean this up
    def get_render_fps(self):
        """Best-effort retrieval of render FPS from env metadata.

        Walks the wrapper chain to find a metadata dict exposing `render_fps`.
        If not found on the env objects, falls back to the loaded spec file
        (if available). Returns an integer FPS when available, otherwise None.
        """
        current = self
        while isinstance(current, gym.Env):
            md = getattr(current, "metadata", None)
            if isinstance(md, dict):
                fps = md.get("render_fps")
                if isinstance(fps, (int, float)) and fps > 0:
                    return int(fps)
            if not hasattr(current, "env"):
                break
            current = getattr(current, "env")
        # Fallback: try spec file
        try:
            spec = self.get_spec()
            fps = spec.get("render_fps", None)
            if isinstance(fps, (int, float)) and fps > 0:
                return int(fps)
        except Exception:
            pass
        return None

    # NOTE: required by ObservationWrapper
    def observation(self, observation): 
        return observation

    def _find_wrapper(self, wrapper_class):
        """Return the first wrapper instance matching `wrapper_class` in the chain.

        Traverses inward via successive `.env` attributes until it finds an
        instance of `wrapper_class` or reaches the base env. Returns None when
        not found.
        """
        current = self
        while True:
            if isinstance(current, wrapper_class): return current
            if not hasattr(current, "env"): return None
            current = getattr(current, "env")

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env_info = EnvInfoWrapper(env)
    print(env_info.get_spec())
    print(env_info.get_reward_treshold())
