from collections.abc import Mapping, Sequence
import json
from typing import Any, Dict

import gymnasium as gym

from utils.decorators import cache
from utils.env_spec import EnvSpec


def _sanitize_spec_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {key: _sanitize_spec_value(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_spec_value(v) for v in value]
    return str(value)


def _env_spec_to_mapping(spec: Any) -> Dict[str, Any]:
    if spec is None:
        return {}
    if isinstance(spec, EnvSpec):
        return spec.as_dict()
    if isinstance(spec, Mapping):
        return dict(spec)

    # Try structured conversions for gymnasium.EnvSpec
    if hasattr(spec, "to_json"):
        try:
            parsed = json.loads(spec.to_json())
            if isinstance(parsed, dict):
                return {k: _sanitize_spec_value(v) for k, v in parsed.items()}
        except Exception:
            pass

    spec_dict: Dict[str, Any] = {}
    for attr in dir(spec):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(spec, attr)
        except Exception:
            continue
        if callable(value):
            continue
        spec_dict[attr] = _sanitize_spec_value(value)
    return spec_dict

# TODO: extract to reusable location
def deep_merge(a: dict, b: dict) -> dict:
    """Return a recursive merge of ``b`` into ``a`` without mutating inputs."""

    result = a.copy()
    for key, value in b.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class GymEnvInfoCollector:
    """Single-pass grab of environment-driven metadata for EnvInfoWrapper."""

    def __init__(self, env: gym.Env):
        self._env = env

    @cache
    def _get_wrappers(self) -> tuple[gym.Env, ...]:
        wrappers: list[gym.Env] = []
        current = self._env
        while isinstance(current, gym.Env):
            wrappers.append(current)
            if not hasattr(current, "env"): break
            current = getattr(current, "env")
        return tuple(wrappers)

    @cache
    def _get_root_env(self) -> gym.Env:
        wrappers = self._get_wrappers()
        assert wrappers, "Expected at least one env wrapper"
        return wrappers[-1]

    def _get_from_wrappers(self, attr: str) -> Any:
        for wrapper in self._get_wrappers():
            if not hasattr(wrapper, attr): continue
            return getattr(wrapper, attr)

    @cache
    def spec_mapping(self) -> Dict[str, Any]:
        root_env = self._get_root_env()
        spec_obj = getattr(root_env, "spec", None)
        return _env_spec_to_mapping(spec_obj)

    @cache
    def get_render_mode(self) -> Any:
        return self._get_from_wrappers("render_mode")

    @cache
    def get_return_threshold(self) -> Any:
        return self._get_from_wrappers("return_threshold")

    @cache
    def get_max_episode_steps(self) -> Any:
        return self._get_from_wrappers("max_episode_steps")

    @cache
    def metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        wrappers = self._get_wrappers()
        for wrapper in reversed(wrappers):
            wrapper_metadata = getattr(wrapper, "metadata", None)
            if isinstance(wrapper_metadata, dict): meta.update(wrapper_metadata)
        return meta

    @cache
    def get_render_fps(self) -> Any:
        default_fps = 30
        metadata = self.metadata()
        if not "render_fps" in metadata: return default_fps
        render_fps = metadata["render_fps"]
        assert isinstance(render_fps, int), "render_fps must be an int"
        return render_fps

    def collect(self) -> Dict[str, Any]:
        return {
            **self.spec_mapping(),
            "render_mode": self.get_render_mode(),
            "return_threshold": self.get_return_threshold(),
            "max_episode_steps": self.get_max_episode_steps(),
            "render_fps": self.get_render_fps(),
        }


class EnvInfoWrapper(gym.ObservationWrapper):

    def __init__(self, env, **kwargs):
        super().__init__(env)

        collected = GymEnvInfoCollector(self).collect()

        # TODO: pass this to spec instead of storing property
        self._obs_type = kwargs.get('obs_type', None)

        override_spec = kwargs.get('spec', {})
        merged_spec = deep_merge(collected, override_spec)

        self._spec = EnvSpec.from_mapping(merged_spec)
    
    def is_rgb_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'rgb'

    def is_ram_env(self):
        obs_type = self.get_obs_type()
        return obs_type == 'ram'
    
    # TODO: proxy unknown methods to spec
    def get_id(self):
        return self._spec.get_id()

    def get_obs_type(self):
        return self._obs_type

    def get_render_mode(self):
        return self._spec.get_render_mode()
        
    def get_return_threshold(self):
        return self._spec.get_return_threshold()

    def get_reward_range(self):
        return self._spec.get_reward_range()
    
    def get_return_range(self):
        return self._spec.get_return_range()

    def get_max_episode_steps(self):
        return self._spec.get_max_episode_steps()

    def get_render_fps(self):
        return self._spec.get_render_fps()

    def get_action_labels(self):
        return self._spec.get_action_labels()

    # NOTE: required by ObservationWrapper
    def observation(self, observation): 
        return observation

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    env_info = EnvInfoWrapper(env)
    print(env_info.get_return_threshold())
