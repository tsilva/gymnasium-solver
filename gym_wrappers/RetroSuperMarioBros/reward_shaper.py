import gymnasium as gym
from gym_wrappers.reward_shaper_base import RewardShaperBase


class RetroSuperMarioBros_RewardShaper(RewardShaperBase):
    """
    Dense reward shaping for Super Mario Bros based on:
    R = V + P + D + S

    Where:
    - V: change in x position (rightward progress, ignoring large resets)
    - P: step penalty (small constant to discourage standing still)
    - D: death penalty or level completion bonus
    - S: change in in-game score

    Total reward is scaled by reward_scale factor.
    """

    def __init__(
        self,
        env,
        reward_scale: float = 0.01,
        death_penalty: float = -50.0,
        level_complete_bonus: float = 50.0,
        step_penalty: float = -0.1,
        x_position_scale: float = 1.0,
        score_scale: float = 0.01,
        x_reset_threshold: float = -100.0,
    ):
        """
        Args:
            env: Environment to wrap
            reward_scale: Global scaling factor for total reward (applied at end)
            death_penalty: Penalty when agent dies
            level_complete_bonus: Bonus when agent completes level
            step_penalty: Small penalty per step to discourage standing still
            x_position_scale: Scale for x position change component
            score_scale: Scale for score change component
            x_reset_threshold: Ignore x deltas more negative than this (level transitions)
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.death_penalty = death_penalty
        self.level_complete_bonus = level_complete_bonus
        self.step_penalty = step_penalty
        self.x_position_scale = x_position_scale
        self.score_scale = score_scale
        self.x_reset_threshold = x_reset_threshold

        # Track previous values
        self.prev_x = None
        self.prev_time = None
        self.prev_score = None
        self.prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize tracking from info dict
        # stable-retro provides RAM variables in info
        self.prev_x = info.get('x', 0) if 'x' in info else info.get('xscrollHi', 0) * 256 + info.get('xscrollLo', 0)
        self.prev_time = info.get('time', 0)
        self.prev_score = info.get('score', 0)
        self.prev_lives = info.get('lives', 3)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract current values from info
        # x position (may be split into high/low bytes in some implementations)
        if 'x' in info:
            current_x = info['x']
        else:
            current_x = info.get('xscrollHi', 0) * 256 + info.get('xscrollLo', 0)

        current_time = info.get('time', 0)
        current_score = info.get('score', 0)
        current_lives = info.get('lives', 3)

        # Initialize components
        v_reward = 0.0  # position change
        p_reward = self.step_penalty  # step penalty (constant)
        d_reward = 0.0  # death/completion
        s_reward = 0.0  # score change

        # V: Change in x position (encourage moving right)
        # Ignore large deltas in both directions (level transitions, death warps, etc.)
        if self.prev_x is not None:
            x_delta = current_x - self.prev_x

            # Filter unrealistic x changes (screen resets, death sequences, etc.)
            if abs(x_delta) > abs(self.x_reset_threshold):
                # Large change detected - ignore to prevent death-farming exploits
                v_reward = 0.0
            else:
                # Normal movement - reward rightward progress
                v_reward = self.x_position_scale * x_delta

        # S: Change in score
        if self.prev_score is not None:
            score_delta = current_score - self.prev_score
            s_reward = self.score_scale * score_delta

        # D: Death penalty or level completion bonus
        # Penalize ANY life loss, not just final death
        if self.prev_lives is not None and current_lives < self.prev_lives:
            # Lost a life (whether mid-episode respawn or final death)
            d_reward = self.death_penalty
        elif terminated:
            # Episode ended without life loss - assume level completion
            d_reward = self.level_complete_bonus

        # Combine all components
        total_shaping = v_reward + p_reward + d_reward + s_reward

        # Scale total reward
        scaled_shaping = self.reward_scale * total_shaping

        # Add debug info
        self._add_shaping_info(
            info,
            scaled_shaping,
            v_reward=v_reward,
            p_reward=p_reward,
            d_reward=d_reward,
            s_reward=s_reward,
            total_unscaled=total_shaping,
        )

        # Update previous values
        self.prev_x = current_x
        self.prev_time = current_time
        self.prev_score = current_score
        self.prev_lives = current_lives

        # Add shaping to original reward
        shaped_reward = scaled_shaping
        #print("Shaped reward: ", shaped_reward)

        return obs, shaped_reward, terminated, truncated, info
