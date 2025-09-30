from typing import Any

from gymnasium.vector import VectorWrapper

# NOTE: once we softcode this wrapper to allow multiple envs, we should look at vec_env_info, apply same pattern and perhaps encapsulate logic somehow
class VecVideoRecorder(VectorWrapper):
    """VectorEnv wrapper that proxies calls to a single underlying EnvVideoRecorder."""

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

        # Fallback: try to call method directly on current env
        method_fn = getattr(env, method)
        return method_fn(*args, **kwargs)

    # ---- Dynamic proxying -------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        assert not name.startswith("_"), f"Cannot proxy private/dunder name: {name}"
        def _proxy_method(*args, **kwargs): return self._call_env(name, *args, **kwargs)
        setattr(self, name, _proxy_method)
        return _proxy_method
