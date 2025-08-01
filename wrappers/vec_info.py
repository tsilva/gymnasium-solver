import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

class VecInfoWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()

    # --- helpers that use self.venv -----------------------------------------

    def _first_base_env(self):
        """
        Try to get the first underlying (unwrapped) gym env from this vec env.
        Works for DummyVecEnv directly; for SubprocVecEnv fall back to get_attr.
        """
        # Walk through nested vec wrappers until we reach something that exposes .envs
        v = self.venv
        while hasattr(v, "venv"):
            if hasattr(v, "envs") and getattr(v, "envs"):
                break
            v = v.venv

        # If we have a concrete list of envs (e.g., DummyVecEnv)
        if hasattr(v, "envs") and getattr(v, "envs"):
            env0 = v.envs[0]
            return getattr(env0, "unwrapped", env0)

        # Otherwise, try the vector API (works for SubprocVecEnv)
        try:
            # ask worker 0 for its unwrapped env (may fail if not picklable)
            return self.venv.get_attr("unwrapped", indices=[0])[0]
        except Exception:
            return None

    # --- public API ----------------------------------------------------------

    def get_spec(self):
        """
        Return an EnvSpec for the first environment if available, else None.
        """
        base = self._first_base_env()

        # Direct read from the base env if we could fetch it
        if base is not None:
            spec = getattr(base, "spec", None)
            if spec is not None:
                return spec
            # last resort via registry if we can find an id
            env_id = getattr(base, "id", None) or getattr(getattr(base, "spec", None), "id", None)
            if env_id:
                try:
                    return gym.spec(env_id)
                except Exception:
                    pass

        # If we couldn't materialize the env object (e.g., Subproc), ask for spec directly
        try:
            spec = self.venv.get_attr("spec", indices=[0])[0]
            if spec is not None:
                return spec
        except Exception:
            pass

        return None

    def get_reward_threshold(self):
        """
        Return the reward_threshold from the env's spec, or None if unavailable.
        """
        spec = self.get_spec()
        return getattr(spec, "reward_threshold", None)