from gymnasium import Env
from typing import Optional
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from gym_wrappers.env_info import EnvInfoWrapper


class VecEnvInfoWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    def _get_nth_env(self, n: int):
        return self.venv.envs[n]

    def _get_env_wrapper(self, env: Env, wrapper_class):
        current_env = env
        while isinstance(current_env, Env):
            if isinstance(current_env, wrapper_class): return current_env
            if not hasattr(current_env, "env"): break
            current_env = current_env.env
        return None

    def _get_env_info_wrapper(self):
        env = self._get_nth_env(0)
        info_wrapper = self._get_env_wrapper(env, EnvInfoWrapper)
        return info_wrapper

    # --- public API ----------------------------------------------------------

    def get_id(self):
        base = self._get_env_info_wrapper()
        return base.get_id()

    def get_spec(self):
        base = self._get_env_info_wrapper()
        return base.get_spec()

    def get_reward_threshold(self):
        base = self._get_env_info_wrapper()
        return base.get_reward_treshold()

    def get_render_fps(self) -> Optional[int]:
        base = self._get_env_info_wrapper()
        return base.get_render_fps()

    def get_obs_type(self):
        base = self._get_env_info_wrapper()
        return base.get_obs_type()

    def is_rgb_env(self):
        base = self._get_env_info_wrapper()
        return base.is_rgb_env()

    def is_ram_env(self):
        base = self._get_env_info_wrapper()
        return base.is_ram_env()