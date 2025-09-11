"""
Thin compatibility shim for vectorized env info.

This VecInfoWrapper does not implement its own logic. It simply proxies
calls to the first underlying env, which is expected to be wrapped with
EnvInfoWrapper by the environment factory. This avoids duplicating logic
and keeps a single source of truth for env metadata.
"""

from typing import Any, Optional

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class VecInfoWrapper(VecEnvWrapper):
    """Proxy info queries to the first underlying `EnvInfoWrapper`.

    Note: We assume each base env was wrapped with `EnvInfoWrapper` before
    vectorization. We therefore forward metadata queries to env index 0.
    """

    # Minimal passthrough for VecEnv protocol
    def reset(self, **kwargs):  # type: ignore[override]
        return self.venv.reset(**kwargs)

    def step_wait(self):  # type: ignore[override]
        return self.venv.step_wait()

    # Internal: resolve the first base env (index 0), unwrapping nested vec wrappers
    def _first_base_env(self) -> Optional[Any]:
        v = self.venv
        # Unwrap nested VecEnvWrapper layers
        while hasattr(v, "venv"):
            v = v.venv
        # Access Dummy/SubprocVecEnv env list when available
        envs = getattr(v, "envs", None)
        if isinstance(envs, (list, tuple)) and len(envs) > 0:
            return envs[0]
        return None

    def _first_env_info(self) -> Optional[Any]:
        base = self._first_base_env()
        # The environment factory should have applied EnvInfoWrapper already
        return base

    # Public proxy methods (delegate to EnvInfoWrapper on env index 0)
    def get_id(self) -> Optional[str]:
        info = self._first_env_info()
        return info.get_id() if info and hasattr(info, "get_id") else None

    def get_spec(self) -> Optional[Any]:
        info = self._first_env_info()
        return info.get_spec() if info and hasattr(info, "get_spec") else None

    def get_max_episode_steps(self) -> Optional[int]:
        info = self._first_env_info()
        return (
            info.get_max_episode_steps()
            if info and hasattr(info, "get_max_episode_steps")
            else None
        )

    def get_time_limit(self) -> Optional[int]:
        info = self._first_env_info()
        return (
            info.get_time_limit()
            if info and hasattr(info, "get_time_limit")
            else None
        )

    # Keep backward compatibility with typo 'treshold' if present on EnvInfoWrapper
    def get_reward_threshold(self) -> Optional[float]:
        info = self._first_env_info()
        if not info:
            return None
        if hasattr(info, "get_reward_threshold"):
            return info.get_reward_threshold()
        if hasattr(info, "get_reward_treshold"):
            return info.get_reward_treshold()
        return None

__all__ = ["VecInfoWrapper"]
