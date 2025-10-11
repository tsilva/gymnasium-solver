import gymnasium as gym
from gym_wrappers.reward_shaper_base import RewardShaperBase


class VizDoom_RewardShaper(RewardShaperBase):
    """
    Lightweight reward shaping for VizDoom scenarios to reduce idling and promote engagement.

    Adds a small negative reward for no-op actions and small positive shaping for
    turning/moving actions. This helps PPO break out of local optima where the
    agent idles and occasionally shoots.

    Additionally supports penalties for wasted ammo (shooting without kills) and health loss.

    The shaping is potential-free (action-based), so it won't dominate true task rewards.
    """

    def __init__(
        self,
        env,
        noop_penalty: float = 0.005,
        move_reward: float = 0.002,
        turn_reward: float = 0.002,
        attack_penalty: float = 0.0,
        ammo_waste_penalty: float = 0.0,
        health_loss_penalty: float = 0.0,
    ):
        super().__init__(env)
        self.noop_penalty = float(noop_penalty)
        self.move_reward = float(move_reward)
        self.turn_reward = float(turn_reward)
        self.attack_penalty = float(attack_penalty)
        self.ammo_waste_penalty = float(ammo_waste_penalty)
        self.health_loss_penalty = float(health_loss_penalty)

        # Track previous values for delta-based penalties
        self._prev_ammo = None
        self._prev_health = None

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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize tracking from first observation
        self._prev_ammo = info.get("ammo", 0.0)
        self._prev_health = info.get("health", 0.0)
        return obs, info

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

        # Penalize ammo waste and health loss
        curr_ammo = info.get("ammo", 0.0)
        curr_health = info.get("health", 0.0)

        if self._prev_ammo is not None and self.ammo_waste_penalty > 0:
            ammo_delta = curr_ammo - self._prev_ammo
            if ammo_delta < 0:  # Used ammo
                # Penalize if no kill reward was earned
                if reward <= 0:
                    shaping -= self.ammo_waste_penalty * abs(ammo_delta)

        if self._prev_health is not None and self.health_loss_penalty > 0:
            health_delta = curr_health - self._prev_health
            if health_delta < 0:  # Lost health
                shaping -= self.health_loss_penalty * abs(health_delta)

        self._prev_ammo = curr_ammo
        self._prev_health = curr_health

        info = dict(info)
        self._add_shaping_info(info, shaping, accumulate=True)

        return obs, reward + shaping, terminated, truncated, info

