from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from utils.decorators import cache
from utils.env_spec import EnvSpec


class EnvInfoWrapper(gym.ObservationWrapper):

    def __init__(self, env, **kwargs):
        super().__init__(env)

        self._obs_type = kwargs.get('obs_type', None)

        _env_spec = self._get_spec__env()
        spec = kwargs.get('spec', {})
        spec = {**_env_spec, **spec}
        self._spec = EnvSpec(spec)

    def get_id(self):
        root_env = self._get_root_env()
        return root_env.spec.id

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @cache
    def get_render_mode(self):
        wrappers = self._collect_wrappers()
        for wrapper in wrappers:
            if hasattr(wrapper, "render_mode"): return wrapper.render_mode
        return None 

    def get_obs_type(self):
        return self._obs_type

    def is_rgb_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'rgb'

    def is_ram_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'ram'

    def get_return_threshold(self):
        spec = self.get_spec()
        return spec.return_threshold()

    def get_reward_range(self):
        spec = self.get_spec()
        return spec.reward_range()

    def get_return_range(self):
        spec = self.get_spec()
        return spec.return_range()

    def _get_time_limit__env(self):
        wrapper = self._find_wrapper(TimeLimit)
        if not wrapper: return None 
        value = getattr(wrapper, "max_episode_steps", None)
        if not value: value = getattr(wrapper, "_max_episode_steps", None)
        if not value: return None
        return value

    def _get_time_limit_spec(self):
        spec = self.get_spec()
        return spec.max_episode_steps()
    
    @cache
    def get_time_limit(self):
        time_limit = self._get_time_limit__env()
        if time_limit: return time_limit

        time_limit = self._get_time_limit_spec()
        if time_limit: return time_limit

        return None

    def _get_render_fps__env(self):
        render_fps = self._get_env_metadata().get("render_fps")
        if not isinstance(render_fps, (int, float)): return None
        if render_fps <= 0: return None
        return int(render_fps)
    
    def _get_render_fps__spec(self):
        spec = self.get_spec()
        return spec.render_fps()
    
    def get_render_fps(self):
        """Best-effort render FPS from env metadata or spec file."""

        # Return FPS from env metadata if available
        fps = self._get_render_fps__env()
        if fps is not None: return fps

        # Return FPS from spec file if available
        fps = self._get_render_fps__spec()
        if fps is not None: return fps

        # Return default FPS if no FPS is available
        return 30   

    def get_action_labels(self):
        spec = self.get_spec()
        return spec.action_labels()

    # NOTE: required by ObservationWrapper
    def observation(self, observation): 
        return observation

    def _get_spec__env(self):
        root_env = self._get_root_env()
        return asdict(root_env.spec)

    @cache
    def _get_root_env(self):
        current = self
        while isinstance(current, gym.Env):
            if not hasattr(current, "env"): break
            current = current.env
        return current

    @cache
    def _find_wrapper(self, wrapper_class):
        wrappers = self._collect_wrappers()
        for wrapper in wrappers:
            if isinstance(wrapper, wrapper_class): return wrapper
        return None

    @cache
    def _collect_wrappers(self):
        wrappers = []
        current = self
        while isinstance(current, gym.Env):
            wrappers.append(current)
            if not hasattr(current, "env"): break
            current = getattr(current, "env")
        return wrappers

    @cache
    def get_spec(self):
        return self._spec

    @cache
    def _get_env_metadata(self):
        wrappers = self._collect_wrappers()
        metadata = {}
        for wrapper in reversed(wrappers):
            wrapper_metadata = getattr(wrapper, "metadata", None)
            if isinstance(wrapper_metadata, dict):
                metadata.update(wrapper_metadata)
        return metadata

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env_info = EnvInfoWrapper(env)
    print(env_info.get_spec())
    print(env_info.get_reward_treshold())
