from typing import Any
from gymnasium.vector import VectorWrapper

class VecEnvInfoWrapper(VectorWrapper):
    """
    VectorEnv helper that proxies unknown method calls to the first underlying env
    (index 0) via call_method. Works for AsyncVectorEnv and SyncVectorEnv.

    Known methods/attributes defined on this wrapper (or parents) behave
    normally. Only *missing* attributes are proxied.
    """

    # ---- VectorEnv overrides -------------------------------------------------
    def reset(self, **kwargs) -> Any:
        return self.env.reset(**kwargs)

    def step(self, actions) -> Any:
        return self.env.step(actions)

    # ---- Internal helper --------------------------------------------------
    def _call_env(self, method: str, *args, **kwargs) -> Any:
        # Navigate through wrappers to find the underlying VectorEnv
        from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

        env = self.env
        visited = set()
        while id(env) not in visited:
            visited.add(id(env))
            # Check if this is an AsyncVectorEnv (has call method)
            if isinstance(env, AsyncVectorEnv):
                # AsyncVectorEnv: use call() to invoke method on first env
                # call() returns a tuple of results from all envs
                results = env.call(method, *args, **kwargs)
                # Return result from first env
                return results[0] if isinstance(results, tuple) else results
            elif isinstance(env, SyncVectorEnv):
                # SyncVectorEnv: access envs list directly
                break
            if not hasattr(env, "env"):
                # Reached the end without finding a VectorEnv
                break
            env = env.env

        # If we found a SyncVectorEnv with envs list, use it
        if isinstance(env, SyncVectorEnv) and hasattr(env, "envs"):
            envs = env.envs
            if isinstance(envs, (list, tuple)) and len(envs) > 0:
                first_env = envs[0]
                # Navigate through this env's wrappers to find the method
                current_env = first_env
                while not hasattr(current_env, method) and hasattr(current_env, "env"):
                    current_env = current_env.env
                method_fn = getattr(current_env, method)
                return method_fn(*args, **kwargs)

        # Check if this is ALE native vectorization (check all envs in the chain)
        is_ale_native = False
        check_env = self.env
        while check_env is not None:
            # Use __dict__ to avoid triggering __getattr__ on wrappers
            if check_env.__dict__.get("_ale_native_vec", False):
                is_ale_native = True
                env = check_env  # Use this env for fallback
                break
            check_env = getattr(check_env, "env", None) if hasattr(check_env, "env") else None

        # ALE native vectorization fallback: provide default implementations
        if is_ale_native:
            return self._ale_native_fallback(env, method, *args, **kwargs)

        # Fallback: try to call method directly on current env
        if hasattr(env, method):
            method_fn = getattr(env, method)
            return method_fn(*args, **kwargs)

        # If method doesn't exist, raise a helpful error
        raise AttributeError(
            f"Method '{method}' not found in vectorized environment. "
            f"This may happen with ALE native vectorization which doesn't support per-env wrappers."
        )

    def _ale_native_fallback(self, env, method: str, *args, **kwargs):
        """Provide fallback implementations for ALE native vectorization."""
        spec = getattr(env, "_spec", None)

        if method == "get_return_threshold":
            if spec and isinstance(spec, dict):
                return spec.get("returns", {}).get("threshold_solved")
            elif spec and hasattr(spec, "get_return_threshold"):
                return spec.get_return_threshold()
            return None
        elif method == "get_max_episode_steps":
            if spec and isinstance(spec, dict):
                return spec.get("time_limit")
            elif spec and hasattr(spec, "get_max_episode_steps"):
                return spec.get_max_episode_steps()
            return None
        elif method == "is_rgb_env":
            obs_type = getattr(env, "_obs_type", None)
            return obs_type == "rgb"
        elif method == "get_id":
            return getattr(env, "env_id", None)
        elif method == "get_spec":
            return spec
        elif method == "get_render_mode":
            return getattr(env, "render_mode", None)
        elif method == "recorder":
            # For ALE native vectorization, try to find the recorder method on the env chain
            # If ALEVecVideoRecorder is present, it will provide the recorder method
            # Otherwise, fall back to no-op context manager
            if hasattr(env, "recorder"):
                # Return the bound method so it can be called with args/kwargs
                return getattr(env, "recorder")(*args, **kwargs)
            from contextlib import nullcontext
            return nullcontext(*args, **kwargs)
        else:
            raise AttributeError(f"No fallback implementation for method '{method}'")

    # ---- Dynamic proxying -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        assert not name.startswith("_"), f"Cannot proxy private/dunder name: {name}"
        def _proxy_method(*args, **kwargs): return self._call_env(name, *args, **kwargs)
        setattr(self, name, _proxy_method)
        return _proxy_method