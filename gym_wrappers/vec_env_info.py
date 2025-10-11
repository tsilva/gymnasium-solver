from typing import Any, Callable
from gymnasium.vector import VectorWrapper

class VecEnvInfoWrapper(VectorWrapper):
    """
    VectorEnv helper that proxies unknown method calls to the first underlying env
    (index 0) via call_method. Works for AsyncVectorEnv and SyncVectorEnv.

    Known methods/attributes defined on this wrapper (or parents) behave
    normally. Only *missing* attributes are proxied.
    """

    def _find_ale_atari_env(self):
        """Check if ALE atari vectorization is used anywhere in chain."""
        env = self.env
        while env is not None:
            if env.__dict__.get("_ale_atari_vec", False):
                return env
            env = getattr(env, "env", None)
        return None

    def _ale_fallbacks(self, env) -> dict[str, Callable]:
        """Map of fallback implementations for ALE native vectorization."""
        spec = getattr(env, "_spec", None)

        def _get_return_threshold_from_spec():
            if spec is None:
                return None
            if hasattr(spec, "get_return_threshold"):
                return spec.get_return_threshold()
            if isinstance(spec, dict):
                returns_cfg = spec.get("returns", {})
                if isinstance(returns_cfg, dict):
                    if "threshold_solved" in returns_cfg:
                        return returns_cfg["threshold_solved"]
                    if "threshold" in returns_cfg:
                        return returns_cfg["threshold"]
            return None

        def _get_max_episode_steps_from_spec():
            if spec is None:
                return None
            if hasattr(spec, "get_max_episode_steps"):
                return spec.get_max_episode_steps()
            if isinstance(spec, dict):
                maybe = spec.get("returns", {})
                if isinstance(maybe, dict):
                    return maybe.get("max_episode_steps") or maybe.get("max_steps")
            return None

        def get_return_threshold():
            threshold = _get_return_threshold_from_spec()
            if threshold is None:
                raise AttributeError("Return threshold not available for ALE vector env")
            return threshold

        def get_max_episode_steps():
            max_steps = _get_max_episode_steps_from_spec()
            if max_steps is None:
                raise AttributeError("Max episode steps not available for ALE vector env")
            return max_steps

        def is_rgb_env():
            return getattr(env, "_obs_type", None) == "rgb"

        def recorder(*args, **kwargs):
            if hasattr(env, "recorder"):
                return env.recorder(*args, **kwargs)
            from contextlib import nullcontext
            return nullcontext(*args, **kwargs)

        return {
            "get_return_threshold": get_return_threshold,
            "get_max_episode_steps": get_max_episode_steps,
            "is_rgb_env": is_rgb_env,
            "get_id": lambda: getattr(env, "env_id", None),
            "get_spec": lambda: spec,
            "get_render_mode": lambda: getattr(env, "render_mode", None),
            "recorder": recorder,
        }

    def _general_fallbacks(self, env) -> dict[str, Callable]:
        """General fallback implementations for common getter methods."""
        spec = env._spec

        def get_spec():
            return spec

        def get_reward_range():
            if hasattr(env, "reward_range"):
                return env.reward_range
            return (-float("inf"), float("inf"))

        def get_action_labels():
            return getattr(env, "_action_labels", None)

        return {
            "get_spec": get_spec,
            "get_reward_range": get_reward_range,
            "get_action_labels": get_action_labels,
        }

    def _call_env(self, method: str, *args, **kwargs) -> Any:
        from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

        # Traverse wrapper chain
        env = self.env
        while env is not None:
            if isinstance(env, AsyncVectorEnv):
                # recorder() cannot work with AsyncVectorEnv because it requires
                # wrapping individual envs in worker processes. Return no-op context manager.
                if method == "recorder":
                    from contextlib import nullcontext
                    return nullcontext()
                results = env.call(method, *args, **kwargs)
                return results[0] if isinstance(results, tuple) else results
            elif isinstance(env, SyncVectorEnv):
                # Found SyncVectorEnv - try first env in list
                assert hasattr(env, "envs") and len(env.envs) > 0
                first_env = env.envs[0]
                while not hasattr(first_env, method) and hasattr(first_env, "env"):
                    first_env = first_env.env
                if hasattr(first_env, method):
                    return getattr(first_env, method)(*args, **kwargs)
                break
            if not hasattr(env, "env"):
                break
            env = env.env

        # Check for ALE atari vectorization
        ale_env = self._find_ale_atari_env()
        if ale_env is not None:
            fallbacks = self._ale_fallbacks(ale_env)
            if method not in fallbacks:
                raise AttributeError(f"No fallback implementation for method '{method}'")
            return fallbacks[method](*args, **kwargs)

        # Try general fallbacks for common getter methods
        #general_fallbacks = self._general_fallbacks(env)
        #if method in general_fallbacks:
       #     return general_fallbacks[method](*args, **kwargs)

        # Direct call fallback
        if not hasattr(env, method):
            raise AttributeError(
                f"Method '{method}' not found in vectorized environment. "
                f"This may happen with ALE atari vectorization which doesn't support per-env wrappers."
            )
        return getattr(env, method)(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        assert not name.startswith("_"), f"Cannot proxy private/dunder name: {name}"
        def _proxy_method(*args, **kwargs): return self._call_env(name, *args, **kwargs)
        setattr(self, name, _proxy_method)
        return _proxy_method
