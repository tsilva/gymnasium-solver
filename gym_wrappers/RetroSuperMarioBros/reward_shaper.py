import gymnasium as gym
from gym_wrappers.reward_shaper_base import RewardShaperBase


class RetroSuperMarioBros_RewardShaper(RewardShaperBase):
    """
    Dense reward shaping for Super Mario Bros based on:
    R = V + T + D + S

    Where:
    - V: change in x position (rightward progress)
    - T: change in game clock (time penalty)
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
        time_penalty_scale: float = 0.1,
        x_position_scale: float = 1.0,
        score_scale: float = 0.01,
    ):
        """
        Args:
            env: Environment to wrap
            reward_scale: Global scaling factor for total reward (applied at end)
            death_penalty: Penalty when agent dies
            level_complete_bonus: Bonus when agent completes level
            time_penalty_scale: Scale for time change component
            x_position_scale: Scale for x position change component
            score_scale: Scale for score change component
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.death_penalty = death_penalty
        self.level_complete_bonus = level_complete_bonus
        self.time_penalty_scale = time_penalty_scale
        self.x_position_scale = x_position_scale
        self.score_scale = score_scale

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
        t_reward = 0.0  # time change
        d_reward = 0.0  # death/completion
        s_reward = 0.0  # score change

        # V: Change in x position (encourage moving right)
        if self.prev_x is not None:
            x_delta = current_x - self.prev_x
            v_reward = self.x_position_scale * x_delta

        # T: Change in game clock (typically counts down, so negative change)
        # Time decreasing is expected, so we apply time_penalty_scale to the delta
        if self.prev_time is not None:
            time_delta = current_time - self.prev_time
            t_reward = self.time_penalty_scale * time_delta

        # S: Change in score
        if self.prev_score is not None:
            score_delta = current_score - self.prev_score
            s_reward = self.score_scale * score_delta

        # D: Death penalty or level completion bonus
        if terminated:
            # Check if death (lives decreased) or level completion
            if self.prev_lives is not None and current_lives < self.prev_lives:
                # Agent died
                d_reward = self.death_penalty
            else:
                # Assume level completion (reached flag)
                d_reward = self.level_complete_bonus

        # Combine all components
        total_shaping = v_reward + t_reward + d_reward + s_reward

        # Scale total reward
        scaled_shaping = self.reward_scale * total_shaping

        # Add debug info
        self._add_shaping_info(
            info,
            scaled_shaping,
            v_reward=v_reward,
            t_reward=t_reward,
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
