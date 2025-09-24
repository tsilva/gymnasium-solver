
import numpy as np
import pytest
import torch

from utils.rollouts import RolloutBuffer, RolloutCollector


class DummyVecEnv:
    def __init__(self, num_envs, obs_dim=4):
        self.num_envs = num_envs
        self._obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
        self._step = 0

    def reset(self):
        self._step = 0
        self._obs.fill(0.0)
        return self._obs.copy()

    def step(self, actions):
        self._step += 1
        next_obs = self._obs + 1.0
        rewards = np.ones(self.num_envs, dtype=np.float32)
        dones = np.array([self._step % 5 == 0] * self.num_envs)
        infos = [{"episode": {"r": 1.0, "l": 5}} if d else {} for d in dones]
        self._obs = next_obs
        return next_obs.copy(), rewards, dones, infos


class DummyPolicy:
    def __init__(self, action_dim=2):
        self.action_dim = action_dim
        self.linear = torch.nn.Linear(4, action_dim)

    def to(self, device):
        return self

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, obs):
        logits = self.linear(obs)
        dist = torch.distributions.Categorical(logits=logits)
        # No value head
        return dist, None


@pytest.mark.unit
def test_rollout_buffer_basic_flatten():
    buf = RolloutBuffer(n_envs=2, obs_shape=(4,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=8)
    start = buf.begin_rollout(4)
    for i in range(4):
        obs_np = np.zeros((2, 4), dtype=np.float32) + i
        next_obs_np = np.zeros((2, 4), dtype=np.float32)
        actions_np = np.zeros(2, dtype=np.int64)
        logps_np = np.zeros(2, dtype=np.float32)
        values_np = np.zeros(2, dtype=np.float32)
        rewards_np = np.zeros(2, dtype=np.float32)
        dones_np = np.zeros(2, dtype=bool)
        timeouts_np = np.zeros(2, dtype=bool)
        buf.add(start + i, obs_np, next_obs_np, actions_np, logps_np, values_np, rewards_np, dones_np, timeouts_np)
    adv = np.zeros((4,2), dtype=np.float32)
    ret = np.zeros((4,2), dtype=np.float32)
    traj = buf.flatten_slice_env_major(start, start + 4, adv, ret)
    assert traj.observations.shape[0] == 8
    assert traj.actions.shape == (8,)


@pytest.mark.unit
def test_rollout_collector_collects_and_computes_metrics():
    env = DummyVecEnv(num_envs=2)
    policy = DummyPolicy()
    collector = RolloutCollector(env, policy, n_steps=6, use_gae=True, normalize_advantages=True)
    traj = collector.collect()
    m = collector.get_metrics()
    assert traj.observations.shape[0] == 12  # n_envs * n_steps
    assert "roll/ep_rew/mean" in m and "roll/fps" in m


@pytest.mark.unit
def test_mc_episode_returns_constant_within_episode():
    class SingleEpisodeEnv:
        def __init__(self, T=5, obs_dim=4):
            self.num_envs = 1
            self._T = int(T)
            self._obs = np.zeros((1, obs_dim), dtype=np.float32)
            self._step = 0

        def reset(self):
            self._step = 0
            self._obs.fill(0.0)
            return self._obs.copy()

        def step(self, actions):
            self._step += 1
            next_obs = self._obs  # keep constant shape/values
            reward = np.array([1.0], dtype=np.float32)
            done = np.array([self._step >= self._T], dtype=bool)
            infos = [
                {"episode": {"r": float(self._T), "l": int(self._T)}} if done[0] else {}
            ]
            self._obs = next_obs
            return next_obs.copy(), reward, done, infos

    env = SingleEpisodeEnv(T=5)
    policy = DummyPolicy()
    # Collect exactly one episode and force MC episode returns
    collector = RolloutCollector(
        env,
        policy,
        n_steps=5,
        use_gae=False,
        normalize_advantages=False,
        gamma=1.0,
        returns_type="episode",
    )
    traj = collector.collect()
    rets = traj.returns.cpu().numpy().reshape(-1)
    # All timesteps within the episode should share the same return (== T)
    assert rets.shape[0] == 5
    np.testing.assert_allclose(rets, np.full_like(rets, rets[0]))
    assert rets[0] == 5.0
