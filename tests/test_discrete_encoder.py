import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gym_wrappers.discrete_encoder import DiscreteEncoder


class DummyDiscreteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n=5):
        super().__init__()
        self.observation_space = spaces.Discrete(n)
        self.action_space = spaces.Discrete(1)
        self._obs = 0

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self._obs = 0
        return self._obs, {}

    def step(self, action):  # type: ignore[override]
        # Cycle observations deterministically for test predictability
        self._obs = (self._obs + 1) % self.observation_space.n
        return self._obs, 0.0, True, False, {}


def test_discrete_encoder_array():
    env = DummyDiscreteEnv(n=5)
    wrapped = DiscreteEncoder(env, encoding="array")
    obs, _ = wrapped.reset()
    assert wrapped.observation_space.shape == (1,)
    assert obs.shape == (1,)
    assert obs.dtype == wrapped.observation_space.dtype
    assert obs[0] == 0


def test_discrete_encoder_binary():
    env = DummyDiscreteEnv(n=5)
    wrapped = DiscreteEncoder(env, encoding="binary")
    obs, _ = wrapped.reset()
    # ceil(log2(5)) == 3
    assert wrapped.observation_space.shape == (3,)
    assert obs.shape == (3,)
    assert np.allclose(obs, np.zeros(3, dtype=np.float32))

    # Directly test encoding of a known value (3 -> 0b011)
    enc = wrapped.observation(3)
    assert np.allclose(enc, np.array([1.0, 1.0, 0.0], dtype=np.float32))


def test_discrete_encoder_onehot():
    env = DummyDiscreteEnv(n=5)
    wrapped = DiscreteEncoder(env, encoding="onehot")
    obs, _ = wrapped.reset()
    assert wrapped.observation_space.shape == (5,)
    assert obs.shape == (5,)
    assert np.allclose(obs, np.array([1, 0, 0, 0, 0], dtype=np.float32))

    enc = wrapped.observation(3)
    assert np.allclose(enc, np.array([0, 0, 0, 1, 0], dtype=np.float32))


def test_discrete_encoder_invalid_space_raises():
    class DummyBoxEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Discrete(1)
        def reset(self, *, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    env = DummyBoxEnv()
    try:
        DiscreteEncoder(env)
        assert False, "Expected ValueError for non-Discrete observation space"
    except ValueError:
        pass
