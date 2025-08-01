
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper, VecEnvStepReturn

class VecNormalizeStatic(VecEnvWrapper):
    def __init__(self, venv: VecEnv):
        super().__init__(venv)
        assert isinstance(venv.observation_space, spaces.Box), "Only supports Box observation spaces."
        self.low = venv.observation_space.low.astype(np.float32)
        self.high = venv.observation_space.high.astype(np.float32)
        self.scale = self.high - self.low
        
        # Update observation space to reflect normalized range
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=venv.observation_space.shape,
            dtype=np.float32,
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) - self.low) / (self.scale + 1e-8)

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        return self._normalize_obs(obs)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._normalize_obs(obs), rewards, dones, infos
