
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorWrapper, VectorEnv
from typing import Tuple, Any


class VecNormalizeStatic(VectorWrapper):
    """Statically normalize observations based on the env's Box bounds.

    - For dimensions with finite and non-degenerate bounds (low < high), map to [0, 1].
    - For dimensions with finite but degenerate bounds (low == high), output a constant 0.0.
    - For dimensions with non-finite bounds (±inf/NaN), pass observations through unchanged.

    This avoids NaNs when environments expose unbounded dimensions (e.g., CartPole
    velocities have ±inf bounds). The wrapped observation_space reflects the mixed
    behavior: [0,1] for normalized dims, {0,0} for degenerate dims, and original
    bounds for unbounded dims.
    """

    def __init__(self, env: VectorEnv):
        super().__init__(env)
        assert isinstance(env.single_observation_space, spaces.Box), "Only supports Box observation spaces."

        # Capture original bounds
        self.low = env.single_observation_space.low.astype(np.float32)
        self.high = env.single_observation_space.high.astype(np.float32)

        # Masks for normalization behavior
        finite = np.isfinite(self.low) & np.isfinite(self.high)
        pos_scale = finite & (self.high > self.low)
        zero_scale = finite & (self.high == self.low)
        self._mask_pos_scale = pos_scale
        self._mask_zero_scale = zero_scale

        # Precompute scale for valid dims; avoid division by zero by construction
        self.scale = np.where(self._mask_pos_scale, (self.high - self.low).astype(np.float32), 1.0)

        # Build the wrapped observation space reflecting the mixed normalization
        low_norm = np.where(self._mask_pos_scale | self._mask_zero_scale, 0.0, self.low).astype(np.float32)
        high_norm = np.where(self._mask_pos_scale, 1.0, np.where(self._mask_zero_scale, 0.0, self.high)).astype(np.float32)
        self.single_observation_space = spaces.Box(low=low_norm, high=high_norm, dtype=np.float32)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32, copy=False)
        # Allocate output and fill per mask to avoid invalid arithmetic with ±inf
        out = np.empty_like(obs, dtype=np.float32)
        # 1) Positive-scale finite dims → [0,1]
        if self._mask_pos_scale.any():
            out[..., self._mask_pos_scale] = (obs[..., self._mask_pos_scale] - self.low[self._mask_pos_scale]) / (
                self.scale[self._mask_pos_scale] + 1e-8
            )
        # 2) Degenerate finite dims (low == high) → constant 0.0
        if self._mask_zero_scale.any():
            out[..., self._mask_zero_scale] = 0.0
        # 3) Non-finite dims → pass-through
        passthrough_mask = ~(self._mask_pos_scale | self._mask_zero_scale)
        if passthrough_mask.any():
            out[..., passthrough_mask] = obs[..., passthrough_mask]
        return out

    def reset(self, **kwargs) -> Tuple[np.ndarray, Any]:
        obs, info = self.env.reset(**kwargs)
        return self._normalize_obs(obs), info

    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._normalize_obs(obs), rewards, terminated, truncated, infos
