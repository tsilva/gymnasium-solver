import numpy as np
import pytest
import torch

from utils.rollouts import RolloutCollector


class DummyVecEnvTimeout1:
    """Single-env VecEnv that emits a truncated done at step 2.

    Timeline (n_steps used in tests = 3):
      t=0: obs=[1], step -> next_obs=[2], reward=0, done=False, info={}
      t=1: obs=[2], step -> next_obs=[0], reward=0, done=True (truncated),
           info={'episode': {'r': 0.0, 'l': 2}, 'TimeLimit.truncated': True, 'terminal_observation': [42.0]}
      t=2: obs=[0], step -> next_obs=[0], reward=0, done=False, info={}
    """

    def __init__(self):
        self.num_envs = 1
        self._step = 0
        self._obs = np.array([[1.0]], dtype=np.float32)  # shape (1,1)

    def reset(self):
        self._step = 0
        self._obs[:] = 1.0
        return self._obs.copy()

    def step(self, actions):
        self._step += 1
        if self._step == 1:
            next_obs = np.array([[2.0]], dtype=np.float32)
            rewards = np.array([0.0], dtype=np.float32)
            dones = np.array([False], dtype=bool)
            infos = [{}]
            self._obs = next_obs
            return next_obs.copy(), rewards, dones, infos
        elif self._step == 2:
            next_obs = np.array([[0.0]], dtype=np.float32)  # new episode start
            rewards = np.array([0.0], dtype=np.float32)
            dones = np.array([True], dtype=bool)
            infos = [
                {
                    'episode': {'r': 0.0, 'l': 2},
                    'TimeLimit.truncated': True,
                    'terminal_observation': np.array([42.0], dtype=np.float32),
                }
            ]
            self._obs = next_obs
            return next_obs.copy(), rewards, dones, infos
        else:
            next_obs = np.array([[0.0]], dtype=np.float32)
            rewards = np.array([0.0], dtype=np.float32)
            dones = np.array([False], dtype=bool)
            infos = [{}]
            self._obs = next_obs
            return next_obs.copy(), rewards, dones, infos


class DeterministicPolicy(torch.nn.Module):
    """Policy that returns fixed actions, zero values in act(), and value=10*mean(obs_features) in predict_values()."""

    def __init__(self):
        super().__init__()
        # Dummy parameter so _device_of can find a device without StopIteration
        self.dummy = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def device(self):
        return self.dummy.device

    def act(self, obs, deterministic=False):
        # Always choose action 0 deterministically
        batch = obs.shape[0]
        a = torch.zeros(batch, dtype=torch.int64, device=obs.device)
        logp = torch.zeros(batch, dtype=torch.float32, device=obs.device)
        v = torch.zeros(batch, dtype=torch.float32, device=obs.device)
        return a, logp, v

    def predict_values(self, obs):
        # Reduce any observation shape to a per-sample scalar by mean over features
        flattened = obs.reshape(obs.shape[0], -1).mean(dim=1)
        val = 10.0 * flattened
        return val.to(dtype=torch.float32)


@pytest.mark.unit
def test_rollout_collector_gae_with_timeout_bootstrap_influences_advantages():
    env = DummyVecEnvTimeout1()
    policy = DeterministicPolicy()

    # Use default gamma=0.99, gae_lambda=0.95 and disable normalization for direct comparison
    collector = RolloutCollector(env, policy, n_steps=3, use_gae=True, normalize_advantages=False)

    traj = collector.collect()

    # With n_envs=1 and T=3, we expect advantages (env-major) to be:
    # t2: 0
    # t1: gamma * 420
    # t0: gamma^2 * lambda * 420
    gamma = collector.gamma
    lam = collector.gae_lambda
    expected = np.array([gamma * gamma * lam * 420.0, gamma * 420.0, 0.0], dtype=np.float32)

    adv = traj.advantages.cpu().numpy().reshape(-1)
    np.testing.assert_allclose(adv, expected, rtol=1e-5, atol=1e-5)

    # returns = advantages + values, and values during act() were zeros
    rets = traj.returns.cpu().numpy().reshape(-1)
    np.testing.assert_allclose(rets, expected, rtol=1e-5, atol=1e-5)

    # Metrics updated
    m = collector.get_metrics()
    assert m['rollout_timesteps'] == 3 * env.num_envs
    assert m['rollout_episodes'] == 1
    assert len(collector.episode_reward_deque) == 1
    assert len(collector.env_episode_reward_deques[0]) == 1
    # Immediate episode metrics captured
    assert 'ep_rew_last' in m and 'ep_len_last' in m
    assert m['ep_rew_last'] == 0.0
    assert m['ep_len_last'] == 2


@pytest.mark.unit
def test_rollout_collector_deterministic_actions_and_shapes():
    class SimpleVecEnv:
        def __init__(self, num_envs=2, obs_dim=4):
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
            dones = np.array([False] * self.num_envs, dtype=bool)
            infos = [{} for _ in range(self.num_envs)]
            self._obs = next_obs
            return next_obs.copy(), rewards, dones, infos

    env = SimpleVecEnv(num_envs=2)
    policy = DeterministicPolicy()
    collector = RolloutCollector(env, policy, n_steps=4, use_gae=True, normalize_advantages=True)

    traj = collector.collect(deterministic=True)

    # Shapes consistent: observations (n_envs*n_steps, obs_dim)
    assert traj.observations.shape == (env.num_envs * 4, 4)
    assert traj.actions.shape == (env.num_envs * 4,)
    # DeterministicPolicy always picks action 0
    np.testing.assert_array_equal(traj.actions.cpu().numpy(), np.zeros(env.num_envs * 4, dtype=np.int64))

    # Metrics presence
    m = collector.get_metrics()
    for k in [
        'total_timesteps', 'total_episodes', 'total_rollouts', 'rollout_timesteps', 'rollout_episodes',
    'ep_rew_last', 'ep_len_last', 'ep_rew_mean', 'ep_len_mean', 'reward_mean', 'reward_std', 'obs_mean', 'obs_std'
    ]:
        assert k in m
