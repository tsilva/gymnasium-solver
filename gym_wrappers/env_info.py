from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

# TODO: extract to utils
from functools import wraps
def cache(method):
    cache = {}
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        key = (method.__name__, args, frozenset(kwargs.items()))
        if key in cache: return cache[key]
        result = method(self, *args, **kwargs)
        cache[key] = result
        return result
    return wrapper

# TODO: create envspec dataclass, allows querying spec map

class EnvInfoWrapper(gym.ObservationWrapper):

    def __init__(self, env, **kwargs):
        super().__init__(env)
        
        self._obs_type = kwargs.get('obs_type', None)
        self._spec = kwargs.get('spec', {})

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
        returns = spec.get('returns', {})
        treshold_solved = returns.get('threshold_solved', None)
        return treshold_solved

    def get_reward_range(self):
        spec = self.get_spec()
        rewards = spec.get('rewards', {})
        rng = rewards.get('range', None)
        if rng is None: return None
        if not isinstance(rng, (list, tuple)): raise ValueError(f"Reward range must be a list or tuple, got {type(rng)}")   
        if len(rng) != 2: raise ValueError(f"Reward range must be a 2-element list or tuple, got {len(rng)}")
        return list(rng)

    def get_return_range(self):
        spec = self.get_spec()
        rewards = spec.get('returns', {})
        rng = rewards.get('range', None)
        if rng is None: return None
        if not isinstance(rng, (list, tuple)): raise ValueError(f"Return range must be a list or tuple, got {type(rng)}")   
        if len(rng) != 2: raise ValueError(f"Return range must be a 2-element list or tuple, got {len(rng)}")
        return list(rng)

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
    
    def _get_render_fps__env(self):
        render_fps = self._get_env_metadata().get("render_fps")
        if not isinstance(render_fps, (int, float)): return None
        if render_fps <= 0: return None
        return int(render_fps)
    
    def _get_render_fps__spec(self):
        spec = self.get_spec()
        render_fps = spec.get("render_fps", None)
        if not isinstance(render_fps, (int, float)): return None
        if render_fps <= 0: return None
        return int(render_fps)
    
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
        action_space = spec.get("action_space", {})
        labels = action_space.get("labels", {})
        if not isinstance(labels, dict): return {}
        return labels

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
        _env_spec = self._get_spec__env()
        spec = {**_env_spec, **self._spec}
        return spec

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
