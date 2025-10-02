import gymnasium as gym
import numpy as np
from gym_wrappers.reward_shaper_base import RewardShaperBase


class MountainCarV0_StateCountBonus(RewardShaperBase):
    """Curiosity-driven exploration bonus for MountainCar-v0 based on state visitation counts.

    Discretizes the continuous state space into bins and rewards visiting rare states
    with an exploration bonus inversely proportional to visit counts.
    """

    def __init__(
        self,
        env,
        position_bins=50,
        velocity_bins=50,
        bonus_scale=1.0,
        bonus_type="count",
        min_count=1,
    ):
        """
        Args:
            env: The environment to wrap
            position_bins: Number of bins for position dimension
            velocity_bins: Number of bins for velocity dimension
            bonus_scale: Scale factor for exploration bonus
            bonus_type: Type of bonus calculation:
                - "count": 1/sqrt(count) scaling
                - "inverse": 1/count scaling
                - "log": 1/log(count + 1) scaling
            min_count: Minimum count to use in denominator (prevents division by zero)
        """
        super().__init__(env)
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.bonus_scale = bonus_scale
        self.bonus_type = bonus_type
        self.min_count = min_count

        # MountainCar state space bounds
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_velocity = -0.07
        self.max_velocity = 0.07

        # Initialize state visit counts
        self.state_counts = np.zeros((position_bins, velocity_bins), dtype=np.int64)
        self.total_steps = 0

    def _discretize_state(self, position, velocity):
        """Convert continuous state to discrete bin indices."""
        # Clip to bounds and normalize to [0, 1]
        pos_norm = np.clip(
            (position - self.min_position) / (self.max_position - self.min_position),
            0.0, 0.999999
        )
        vel_norm = np.clip(
            (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity),
            0.0, 0.999999
        )

        # Convert to bin indices
        pos_bin = int(pos_norm * self.position_bins)
        vel_bin = int(vel_norm * self.velocity_bins)

        return pos_bin, vel_bin

    def _compute_bonus(self, count):
        """Compute exploration bonus based on visit count."""
        # Use min_count to prevent division by zero
        effective_count = max(count, self.min_count)

        if self.bonus_type == "count":
            # 1/sqrt(count) bonus - standard choice in literature
            bonus = 1.0 / np.sqrt(effective_count)
        elif self.bonus_type == "inverse":
            # 1/count bonus - stronger emphasis on novelty
            bonus = 1.0 / effective_count
        elif self.bonus_type == "log":
            # 1/log(count+1) bonus - slower decay
            bonus = 1.0 / np.log(effective_count + 1)
        else:
            raise ValueError(f"Unknown bonus_type: {self.bonus_type}")

        return self.bonus_scale * bonus

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Don't reset state counts on episode boundary - maintain across episodes
        # This encourages exploration across the entire training run

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs

        # Discretize current state
        pos_bin, vel_bin = self._discretize_state(position, velocity)

        # Get current visit count for this state
        count = self.state_counts[pos_bin, vel_bin]

        # Compute exploration bonus before incrementing count
        exploration_bonus = self._compute_bonus(count)

        # Increment visit count
        self.state_counts[pos_bin, vel_bin] += 1
        self.total_steps += 1

        # Add debug info
        self._add_shaping_info(
            info,
            exploration_bonus,
            exploration_bonus=exploration_bonus,
            state_visit_count=count + 1,
            unique_states_visited=np.count_nonzero(self.state_counts),
            total_states=self.position_bins * self.velocity_bins,
        )

        # Add exploration bonus to original reward
        shaped_reward = reward + exploration_bonus

        return obs, shaped_reward, terminated, truncated, info
