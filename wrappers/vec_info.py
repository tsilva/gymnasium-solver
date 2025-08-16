import gymnasium as gym
import numpy as np
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
        if spec is not None:
            return getattr(spec, "reward_threshold", None)
            
        # Fallback: If we can't get spec from the vectorized environment,
        # try to create a single instance to get the environment spec
        try:
            # Try to get env_id from the first environment
            base = self._first_base_env()
            if base is not None:
                env_id = getattr(base, "id", None) or getattr(getattr(base, "spec", None), "id", None)
                if env_id:
                    import gymnasium as gym
                    # For Atari environments, need to register ALE
                    if env_id.startswith("ALE/"):
                        try:
                            import ale_py
                            gym.register_envs(ale_py)
                        except ImportError:
                            pass
                    
                    temp_spec = gym.spec(env_id)
                    return getattr(temp_spec, "reward_threshold", None)
        except Exception:
            pass
            
        return None

    def get_reward_range(self):
        """Return the reward_range from the underlying environment if available.

        Attempts the following, in order:
        - Direct attribute on first base env (DummyVecEnv path)
        - get_attr('reward_range') on the VecEnv (SubprocVecEnv path)
        - Fallback to Gymnasium registry by resolving spec and instantiating a temp env
        """
        # Try reading from the first underlying base env
        try:
            base = self._first_base_env()
            if base is not None and hasattr(base, "reward_range"):
                rr = getattr(base, "reward_range")
                # ensure it is a tuple-like of length 2
                if isinstance(rr, (tuple, list)) and len(rr) == 2:
                    return tuple(rr)
        except Exception:
            pass

        # Try vec env API (works for SubprocVecEnv)
        try:
            rr = self.venv.get_attr("reward_range", indices=[0])[0]
            if isinstance(rr, (tuple, list)) and len(rr) == 2:
                return tuple(rr)
        except Exception:
            pass

        # Last resort: resolve via env spec and a temporary instance
        try:
            base = self._first_base_env()
            env_id = None
            if base is not None:
                env_id = getattr(base, "id", None) or getattr(getattr(base, "spec", None), "id", None)
            if env_id:
                import gymnasium as gym
                if env_id.startswith("ALE/"):
                    try:
                        import ale_py
                        gym.register_envs(ale_py)
                    except Exception:
                        pass
                # Instantiate a lightweight temp env to read reward_range
                try:
                    tmp_env = gym.make(env_id)
                    rr = getattr(tmp_env, "reward_range", None)
                    try:
                        tmp_env.close()
                    except Exception:
                        pass
                    if isinstance(rr, (tuple, list)) and len(rr) == 2:
                        return tuple(rr)
                except Exception:
                    pass
        except Exception:
            pass

        return None
    
    def get_input_dim(self):
        """
        Return a reasonable flat input dimension for the observation space.

        - Discrete: return 1 (use scalar ID as input)
        - Box: return first dimension (assumes flat features or 1D vector)
        - MultiDiscrete: return number of subspaces
        - Tuple: sum of subspace flat dims when possible
        """
        from gymnasium import spaces

        obs_space = self.venv.observation_space

        # Discrete spaces (e.g., Taxi-v3): use scalar state id as a single feature
        if isinstance(obs_space, spaces.Discrete):
            return 1

        # Box spaces: favor 1D vectors; otherwise return first dim as a heuristic
        if isinstance(obs_space, spaces.Box):
            if len(obs_space.shape) == 1:
                return int(obs_space.shape[0])
            # For 2D/3D observations, fallback to product if small, else first dim
            try:
                prod = int(1)
                for s in obs_space.shape:
                    prod *= int(s)
                # Avoid very large flatten sizes silently; caller may choose CNN
                return prod
            except Exception:
                return int(obs_space.shape[0]) if obs_space.shape else None

        # MultiDiscrete: number of discrete components
        if isinstance(obs_space, spaces.MultiDiscrete):
            return int(obs_space.nvec.size)

        # Tuple: sum of component flat dims when possible
        if isinstance(obs_space, spaces.Tuple):
            dims = []
            for s in obs_space.spaces:
                if isinstance(s, spaces.Discrete):
                    dims.append(1)
                elif hasattr(s, 'shape') and s.shape:
                    dims.append(int(np.prod(s.shape)))
                else:
                    dims.append(0)
            return int(sum(dims)) if dims else None

        # Fallback
        if hasattr(obs_space, 'shape') and obs_space.shape:
            return int(obs_space.shape[0])
        return None

    def get_output_dim(self):
        """
        Return the output dimension (action space size).
        """
        action_space = self.venv.action_space
        if hasattr(action_space, 'n'):  # Discrete action space
            return action_space.n
        elif hasattr(action_space, 'shape') and action_space.shape:  # Continuous action space
            return action_space.shape[0]
        return None