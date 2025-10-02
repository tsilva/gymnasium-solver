import gymnasium as gym
import numpy as np
from gym_wrappers.reward_shaper_base import RewardShaperBase


class CartPoleV1_RewardShaper(RewardShaperBase):
    """
    Reward shaping for CartPole-v1 focusing on keeping:
    - the cart near the center (x ≈ 0)
    - the pole near upright (theta ≈ 0)

    Uses potential-based shaping so learning is accelerated while preserving optimal policies.

    Phi(s) = w_angle * (1 - |theta|/theta_thresh) + w_pos * (1 - |x|/x_thresh)
    Shaping R_s = (Phi(s') - Phi(s))

    Notes:
    - theta is in radians, theta_thresh ≈ 12° = ~0.20944 rad in CartPole-v1
    - x_thresh ≈ 2.4
    - You can optionally add small velocity damping terms if desired in the future.
    """

    def __init__(
        self,
        env,
        angle_reward_scale: float = 1.0,
        position_reward_scale: float = 0.25,
        clip_potential: bool = True,
    ):
        super().__init__(env)
        self.angle_reward_scale = float(angle_reward_scale)
        self.position_reward_scale = float(position_reward_scale)
        self.clip_potential = bool(clip_potential)

        # Default CartPole thresholds
        # Done when |x| > x_threshold or |theta| > theta_threshold_radians
        # Values taken from classic control CartPole implementation
        self.x_threshold = getattr(self.env.unwrapped, "x_threshold", 2.4)
        self.theta_threshold = getattr(self.env.unwrapped, "theta_threshold_radians", np.deg2rad(12))

        self._prev_phi = None

    def _phi(self, obs):
        # obs = [x, x_dot, theta, theta_dot]
        x = float(obs[0])
        theta = float(obs[2])

        # Normalize distances to [0, 1] where 1 is perfect (center/upright)
        pos_term = 1.0 - abs(x) / max(self.x_threshold, 1e-6)
        angle_term = 1.0 - abs(theta) / max(self.theta_threshold, 1e-6)

        if self.clip_potential:
            pos_term = float(np.clip(pos_term, 0.0, 1.0))
            angle_term = float(np.clip(angle_term, 0.0, 1.0))

        phi = self.angle_reward_scale * angle_term + self.position_reward_scale * pos_term
        return float(phi)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_phi = self._phi(obs)
        shaping = 0.0
        if self._prev_phi is not None:
            shaping = curr_phi - self._prev_phi

        # Expose shaping components for debugging/analysis
        self._add_shaping_info(info, shaping, potential_prev=self._prev_phi, potential_curr=curr_phi)

        self._prev_phi = curr_phi

        return obs, reward + shaping, terminated, truncated, info
