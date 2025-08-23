import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
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
        Works for DummyVecEnv directly; for SubprocVecEnv we avoid fetching
        the env instance over IPC because it may not be picklable (e.g.,
        ALEInterface). In that case, return None and let callers use
        picklable attributes via get_attr instead.
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

        # For SubprocVecEnv (no local .envs), do not attempt to fetch
        # the env object via get_attr("unwrapped") as it can contain
        # non-picklable handles. Return None to signal unavailable.
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
        # 1) Try env spec
        spec = self.get_spec()
        if spec is not None:
            thr = getattr(spec, "reward_threshold", None)
            if thr is not None:
                return thr

        # 2) Try complementary env info YAML
        yaml_thr = self._get_complement_threshold()
        if yaml_thr is not None:
            return yaml_thr
            
        # 3) Fallback: If we can't get spec from the vectorized environment,
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

        # Try complementary env info YAML
        yaml_rr = self._get_complement_reward_range()
        if isinstance(yaml_rr, (tuple, list)) and len(yaml_rr) == 2:
            return tuple(yaml_rr)

        # Last resort: resolve via env spec and a temporary instance
        try:
            env_id = None
            spec = self.get_spec()
            if spec is not None:
                env_id = getattr(spec, "id", None)
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
    
    # --- complementary info (YAML) ------------------------------------------
    def _get_env_id(self) -> Optional[str]:
        """Best-effort retrieval of the base environment id (e.g., 'CartPole-v1')."""
        try:
            base = self._first_base_env()
            if base is not None:
                env_id = getattr(base, "id", None) or getattr(getattr(base, "spec", None), "id", None)
                if env_id:
                    return str(env_id)
        except Exception:
            pass
        try:
            spec = self.get_spec()
            if spec is not None and getattr(spec, "id", None):
                return str(spec.id)
        except Exception:
            pass
        # Fallbacks: try attributes exposed on the vectorized env itself
        try:
            venv_env_id = getattr(self.venv, "env_id", None)
            if venv_env_id:
                return str(venv_env_id)
        except Exception:
            pass
        try:
            # If running under SubprocVecEnv, attempt IPC-safe attribute fetch
            env_id_attr = self.venv.get_attr("env_id", indices=[0])[0]
            if env_id_attr:
                return str(env_id_attr)
        except Exception:
            pass
        return None

    def _load_complement_yaml(self) -> Optional[Dict[str, Any]]:
        """
        Load complementary env info YAML by env_id.

        Search order:
          1) Path from ENV_INFO_DIR environment variable
          2) Project default: <project_root>/config/env_info

        File name is '<env_id>.yaml'. Returns parsed dict or None.
        """
        env_id = self._get_env_id()
        if not env_id:
            return None

        # Determine candidate directories
        candidates = []
        custom_dir = os.environ.get("ENV_INFO_DIR")
        if custom_dir:
            candidates.append(Path(custom_dir))
        try:
            # Project root: two parents up from this file (gym_wrappers/vec_info.py)
            project_root = Path(__file__).resolve().parents[1]
            candidates.append(project_root / "config" / "env_info")
        except Exception:
            pass

        # Try loading YAML from candidates
        for base_dir in candidates:
            try:
                path = base_dir / f"{env_id}.yaml"
                if path.is_file():
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    if isinstance(data, dict):
                        return data
            except Exception:
                continue
        return None

    def _get_complement_threshold(self) -> Optional[float]:
        data = self._load_complement_yaml()
        if not isinstance(data, dict):
            return None

        # Preferred: explicit top-level reward_threshold
        candidates: list[Any] = [data.get("reward_threshold")]

        # Backwards-compatible fallbacks from the env_info schema used in this repo
        #   rewards:
        #     threshold: <float>
        #     threshold_solved: <float>
        try:
            rewards_block = data.get("rewards") or {}
            if isinstance(rewards_block, dict):
                candidates.append(rewards_block.get("threshold"))
                candidates.append(rewards_block.get("threshold_solved"))
        except Exception:
            pass

        # Return the first candidate that can be parsed as a float
        for val in candidates:
            if val is None:
                continue
            try:
                return float(val)
            except Exception:
                continue

        return None

    def _get_complement_reward_range(self) -> Optional[Tuple[float, float]]:
        data = self._load_complement_yaml()
        if isinstance(data, dict):
            rr = data.get("reward_range")
            if isinstance(rr, (list, tuple)) and len(rr) == 2:
                try:
                    lo = float(rr[0]) if rr[0] is not None else None
                    hi = float(rr[1]) if rr[1] is not None else None
                    if lo is not None and hi is not None:
                        return (lo, hi)
                except Exception:
                    return None
        return None
    
    def _get_complement_action_labels(self) -> Optional[list]:
        """
        Load ordered action labels from env_info YAML if present.
        Expected structure:
          action_space:
            discrete: <int>
            labels: {0: "NOOP", 1: "FIRE", ...}
        Returns a list of strings of length `discrete`, or None if unavailable.
        """
        data = self._load_complement_yaml()
        if not isinstance(data, dict):
            return None
        try:
            action_space = data.get("action_space") or {}
            discrete = action_space.get("discrete")
            labels = action_space.get("labels")
            if isinstance(labels, list) and isinstance(discrete, int) and len(labels) == int(discrete):
                # Already ordered list
                return [str(x) for x in labels]
            if not isinstance(discrete, int) or discrete <= 0 or not isinstance(labels, dict):
                return None
            ordered: list = []
            for i in range(int(discrete)):
                label = labels.get(i)
                if not isinstance(label, str):
                    return None
                ordered.append(label)
            return ordered
        except Exception:
            return None

    def get_action_labels(self) -> Optional[list]:
        """
        Attempt to retrieve ordered action labels for discrete action spaces.
        Currently sourced from complementary env_info YAML if available.
        """
        # Only makes sense for discrete action spaces
        try:
            if hasattr(self.venv.action_space, "n"):
                labels = self._get_complement_action_labels()
                if isinstance(labels, list) and len(labels) == int(self.venv.action_space.n):
                    return labels
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
    
    def print_spec(self):
        # Observation space and action space from vectorized env
        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

        # Reward range and threshold when available
        reward_range = self.get_reward_range()
        print(f"Reward range: {reward_range}")

        # Reward threshold if defined
        reward_threshold = self.get_reward_threshold()
        print(f"Reward threshold: {reward_threshold}")
        