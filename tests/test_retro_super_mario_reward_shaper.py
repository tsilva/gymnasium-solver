import gymnasium as gym
import numpy as np
import pytest

from gym_wrappers.RetroSuperMarioBros.reward_shaper import RetroSuperMarioBros_RewardShaper


class _DummyMarioEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self._step = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(2, 2, 3),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        del kwargs
        self._step = 0
        obs = np.zeros((2, 2, 3), dtype=np.uint8)
        info = {"x": 10, "time": 400, "score": 0, "lives": 2}
        return obs, info

    def step(self, action):
        del action
        self._step += 1
        obs = np.zeros((2, 2, 3), dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"x": 12, "time": 399, "score": 100, "lives": 2}
        return obs, reward, terminated, truncated, info


@pytest.mark.unit
def test_reward_shaper_does_not_emit_debug_info_by_default():
    env = RetroSuperMarioBros_RewardShaper(_DummyMarioEnv(), reward_scale=1.0)
    env.reset()
    _obs, reward, _terminated, _truncated, info = env.step(0)

    assert reward != 0.0
    assert "shaping_reward" not in info
    assert "total_unscaled" not in info


@pytest.mark.unit
def test_reward_shaper_can_emit_debug_info_when_requested():
    env = RetroSuperMarioBros_RewardShaper(
        _DummyMarioEnv(),
        reward_scale=1.0,
        emit_debug_info=True,
    )
    env.reset()
    _obs, reward, _terminated, _truncated, info = env.step(0)

    assert info["shaping_reward"] == reward
    assert "total_unscaled" in info
