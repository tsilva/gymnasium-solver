from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from gym_wrappers.utils import find_wrapper
from utils.io import read_yaml

# TODO: CLEANUP this file

class EnvInfoWrapper(gym.ObservationWrapper):

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._obs_type = kwargs.get('obs_type', None)
        # Optional: project/challenge id (YAML filename stem) to prefer when resolving specs
        self._project_id = kwargs.get('project_id', None)
        self._spec = kwargs.get('spec', None)

    def _get_root_env(self):
        current = self
        while isinstance(current, gym.Env):
            if not hasattr(current, "env"): break
            current = current.env
        return current

    def get_id(self):
        root_env = self._get_root_env()
        return root_env.spec.id

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _get_spec__env(self):
        root_env = self._get_root_env()
        return asdict(root_env.spec)

    def get_spec(self):
        _env_spec = self._get_spec__env()
        spec = {**_env_spec, **self._spec} # TODO: cache
        return spec

    def get_obs_type(self):
        return self._obs_type

    def is_rgb_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'rgb'

    def is_ram_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'ram'

    def get_return_threshold(self):
        """Backward-compatible threshold accessor.

        Prefers returns.threshold_solved when available; falls back to
        rewards.threshold_solved for older specs.
        """
        spec = self.get_spec()
        # New style: prefer returns.threshold_solved
        returns = spec.get('returns', {}) if isinstance(spec, dict) else {}
        if isinstance(returns, dict) and 'threshold_solved' in returns:
            return returns['threshold_solved']
        # Legacy location: rewards.threshold_solved
        rewards = spec.get('rewards', {}) if isinstance(spec, dict) else {}
        if isinstance(rewards, dict) and 'threshold_solved' in rewards:
            return rewards['threshold_solved']
        return None

    # TODO: CLEANUP this method
    # Preferred names for external callers
    def get_reward_range(self):
        """Per-step reward range [min, max] when provided in spec.

        Looks under rewards.range (per-step). Returns None if unavailable.
        """
        try:
            spec = self.get_spec()
            rewards = spec.get('rewards') if isinstance(spec, dict) else None
            if isinstance(rewards, dict):
                rng = rewards.get('range')
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return list(rng)
        except Exception:
            pass
        return None

    def get_return_range(self):
        """Episodic return range [min, max] when provided in spec.

        Prefers returns.range; falls back to legacy rewards.range when returns
        section is missing.
        """
        try:
            spec = self.get_spec()
            returns = spec.get('returns') if isinstance(spec, dict) else None
            if isinstance(returns, dict):
                rng = returns.get('range')
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return list(rng)
            # Fallback to legacy location
            rewards = spec.get('rewards') if isinstance(spec, dict) else None
            if isinstance(rewards, dict):
                rng = rewards.get('range')
                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                    return list(rng)
        except Exception:
            pass
        return None

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
        """Best-effort render FPS from env metadata or spec file."""
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
        return 30

    def get_action_labels(self):
        return self.get_spec().get("action_space", {}).get("labels", {})

    # NOTE: required by ObservationWrapper
    def observation(self, observation): 
        return observation

    def _find_wrapper(self, wrapper_class):
        return find_wrapper(self, wrapper_class)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env_info = EnvInfoWrapper(env)
    print(env_info.get_spec())
    print(env_info.get_reward_treshold())
