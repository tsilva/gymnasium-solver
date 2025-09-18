import gymnasium as gym


class VizDoom_RewardShaper(gym.Wrapper):
    """
    Lightweight reward shaping for VizDoom scenarios to reduce idling and promote engagement.

    Adds a small negative reward for no-op actions and small positive shaping for
    turning/moving actions. This helps PPO break out of local optima where the
    agent idles and occasionally shoots.

    The shaping is potential-free (action-based), so it won't dominate true task rewards.
    """

    def __init__(
        self,
        env,
        noop_penalty: float = 0.005,
        move_reward: float = 0.002,
        turn_reward: float = 0.002,
        attack_penalty: float = 0.0,
    ):
        super().__init__(env)
        self.noop_penalty = float(noop_penalty)
        self.move_reward = float(move_reward)
        self.turn_reward = float(turn_reward)
        self.attack_penalty = float(attack_penalty)

        # Cache action semantics when available (env exposes a discrete action set)
        self._action_meanings = {
            0: "noop",
            1: "forward",
            2: "backward",
            3: "strafe_left",
            4: "strafe_right",
            5: "turn_left",
            6: "turn_right",
            7: "attack",
            8: "forward_attack",
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping = 0.0

        # Encourage movement/turning; discourage pure no-op idling
        if isinstance(self.action_space, gym.spaces.Discrete):
            act = int(action)
            meaning = self._action_meanings.get(act, None)
            if meaning == "noop":
                shaping -= self.noop_penalty
            else:
                if meaning in {"forward", "backward", "strafe_left", "strafe_right", "forward_attack"}:
                    shaping += self.move_reward
                if meaning in {"turn_left", "turn_right"}:
                    shaping += self.turn_reward
                if meaning == "attack":
                    shaping -= self.attack_penalty

        info = dict(info)
        info["shaping_reward"] = info.get("shaping_reward", 0.0) + shaping

        return obs, reward + shaping, terminated, truncated, info

