from typing import Optional

from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


class VecEnvInfoWrapper(VecEnvWrapper):
    """
    VecEnv helper that exposes environment metadata from the first underlying env.

    Works with both DummyVecEnv (has `envs`) and SubprocVecEnv (no direct access)
    by delegating calls via `env_method` to the wrapped single-environment
    EnvInfoWrapper inside each worker.
    """

    # ---- VecEnv overrides -------------------------------------------------
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    # ---- Internal helpers -------------------------------------------------
    def _call_env(self, method: str, *args, default=None, **kwargs):
        """Call a method on env index 0 via VecEnv.env_method.

        This supports SubprocVecEnv (no `.envs`) and DummyVecEnv alike.
        Returns the first result or `default` if unavailable.
        """
        try:
            # env_method returns a list, even for a single index
            result = self.venv.env_method(method, *args, indices=[0], **kwargs)
            if isinstance(result, (list, tuple)) and result:
                return result[0]
            return default
        except Exception:
            # Let higher level errors surface when metadata is critical; otherwise
            # return the provided default for best-effort helpers like fps, etc.
            if default is not None:
                return default
            raise

    # --- public API --------------------------------------------------------
    def get_id(self):
        return self._call_env("get_id")

    def get_spec(self):
        return self._call_env("get_spec")

    def get_reward_threshold(self):
        return self._call_env("get_reward_treshold")

    def get_render_fps(self) -> Optional[int]:
        return self._call_env("get_render_fps", default=None)

    def get_time_limit(self) -> Optional[int]:
        return self._call_env("get_time_limit", default=None)

    def get_obs_type(self):
        return self._call_env("get_obs_type", default=None)

    def is_rgb_env(self):
        return bool(self._call_env("is_rgb_env", default=False))

    def is_ram_env(self):
        return bool(self._call_env("is_ram_env", default=False))
