import gymnasium as gym


class RewardShaperBase(gym.Wrapper):
    """Base class for reward shaping wrappers with common debug info tracking."""

    def _add_shaping_info(self, info: dict, shaping_reward: float, accumulate: bool = False, **components) -> dict:
        """
        Add shaping reward and optional component breakdowns to info dict.

        Args:
            info: The info dict from env.step()
            shaping_reward: Total shaping reward to add
            accumulate: If True, add to existing shaping_reward instead of replacing
            **components: Optional named components (e.g., position_shaping=0.1, velocity_shaping=0.2)

        Returns:
            The modified info dict with shaping debug data added
        """
        if accumulate:
            info['shaping_reward'] = info.get('shaping_reward', 0.0) + shaping_reward
        else:
            info['shaping_reward'] = shaping_reward
        info.update(components)
        return info
