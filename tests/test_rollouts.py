
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

    @torch.inference_mode()
    def act(self, obs, deterministic=False):
        logits = self.linear(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = torch.zeros_like(logp)
        return a, logp, v

    @torch.inference_mode()
    def predict_values(self, obs):
        return torch.zeros(obs.shape[0])


@pytest.mark.unit
def test_rollout_buffer_basic_flatten():
    buf = RolloutBuffer(n_envs=2, obs_shape=(4,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=8)
    start = buf.begin_rollout(4)
    for i in range(4):
        obs_t = torch.zeros((2, 4)) + i
        a_t = torch.zeros(2, dtype=torch.int64)
        lp_t = torch.zeros(2)
        v_t = torch.zeros(2)
        buf.store_tensors(start + i, obs_t, a_t, lp_t, v_t)
        buf.store_cpu_step(start + i, np.zeros((2,4), dtype=np.float32), np.zeros((2,4), dtype=np.float32), np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool))
    buf.copy_tensors_to_cpu(start, start + 4)
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
    assert "ep_rew_mean" in m and "rollout_fps" in m
