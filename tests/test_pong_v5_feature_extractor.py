from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_wrappers.PongV5.feature_extractor import PongV5_FeatureExtractor


class DummyPongEnv(gym.Env):
    """Minimal Pong-like env exposing OCAtari-style objects for wrapper tests."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=np.uint8,
        )
        self.objects = []

    def _build_objects(self):
        # Provide the minimal attributes consumed by the feature extractor helpers.
        return [
            SimpleNamespace(category="Player", h=16.0, dy=0.0, center=(16.0, 100.0)),
            SimpleNamespace(category="Enemy", h=16.0, dy=0.0, center=(144.0, 110.0)),
            SimpleNamespace(
                category="Ball",
                w=2.0,
                h=2.0,
                dx=1.0,
                dy=-1.0,
                visible=True,
                center=(80.0, 120.0),
            ),
        ]

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self.objects = self._build_objects()
        observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return observation, {}

    def step(self, action):  # type: ignore[override]
        # Keep objects deterministic; the feature extractor only depends on their fields.
        self.objects = self._build_objects()
        observation = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info


def _tail(obs, base_dim=9):
    return obs[base_dim:]


def test_last_action_vector_defaults_to_env_action_space():
    env = DummyPongEnv()
    wrapper = PongV5_FeatureExtractor(env)

    obs, _info = wrapper.reset()
    assert obs.shape == (15,)
    assert np.allclose(_tail(obs), 0.0)

    step_obs, *_ = wrapper.step(5)
    expected_tail = np.zeros(6, dtype=np.float32)
    expected_tail[5] = 1.0
    assert np.allclose(_tail(step_obs), expected_tail)


def test_last_action_vector_respects_action_ids_mapping():
    env = DummyPongEnv()
    wrapper = PongV5_FeatureExtractor(env, action_ids=[0, 2, 3])

    obs, _info = wrapper.reset()
    assert obs.shape == (12,)
    assert np.allclose(_tail(obs), 0.0)

    step_obs, *_ = wrapper.step(3)
    expected_tail = np.zeros(3, dtype=np.float32)
    expected_tail[2] = 1.0  # action_ids[2] == 3
    assert np.allclose(_tail(step_obs), expected_tail)
