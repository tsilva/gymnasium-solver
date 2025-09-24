import numpy as np
import pytest
import torch

from utils.rollouts import (
    RollingWindow,
    RolloutCollector,
    RunningStats,
    _build_idx_map_from_valid_mask,
    _build_valid_mask_and_index_map,
    _normalize_advantages,
    compute_batched_mc_returns,
)


@pytest.mark.unit
def test_build_idx_map_from_valid_mask_basic():
    mask = np.array([False, False, True, False, True, False], dtype=bool)
    idx_map = _build_idx_map_from_valid_mask(mask)
    # Before first valid (idx 2) maps to first valid; subsequent invalid map to nearest previous valid
    np.testing.assert_array_equal(idx_map, np.array([2, 2, 2, 2, 4, 4], dtype=np.int64))

    # No valid entries returns None
    assert _build_idx_map_from_valid_mask(np.zeros(5, dtype=bool)) is None


@pytest.mark.unit
def test_build_valid_mask_and_index_map_multi_env():
    T, N = 5, 2
    # Env0 has a real terminal at t=1; Env1 has no terminal in the slice
    dones = np.array([
        [False, False],
        [True,  False],
        [False, False],
        [False, False],
        [False, False],
    ], dtype=bool)
    timeouts = np.zeros((T, N), dtype=bool)

    valid_mask_flat, idx_map = _build_valid_mask_and_index_map(dones, timeouts)
    assert valid_mask_flat is not None and idx_map is not None
    # Env-major flatten: env0 t=0..1 valid, rest invalid
    expected_mask = np.array([True, True, False, False, False, False, False, False, False, False], dtype=bool)
    np.testing.assert_array_equal(valid_mask_flat, expected_mask)
    np.testing.assert_array_equal(idx_map, np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64))

    # Case: no terminals at all -> both None
    vm, im = _build_valid_mask_and_index_map(np.zeros((T, N), dtype=bool), np.zeros((T, N), dtype=bool))
    assert vm is None and im is None


@pytest.mark.unit
def test_mc_returns_with_timeouts_toggle():
    # Single env with a timeout at t=1
    rewards = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    dones = np.array([[False], [True], [False], [False]], dtype=bool)
    timeouts = np.array([[False], [True], [False], [False]], dtype=bool)

    # Treat timeouts as terminals: pass zeros for timeouts
    rets_term = compute_batched_mc_returns(rewards, dones, np.zeros_like(timeouts), gamma=1.0)
    np.testing.assert_array_equal(rets_term.reshape(-1), np.array([3.0, 2.0, 7.0, 4.0], dtype=np.float32))

    # Do not treat timeouts as terminals: use actual timeouts (propagates across timeout)
    rets_noterm = compute_batched_mc_returns(rewards, dones, timeouts, gamma=1.0)
    np.testing.assert_array_equal(rets_noterm.reshape(-1), np.array([10.0, 9.0, 7.0, 4.0], dtype=np.float32))


@pytest.mark.unit
def test_normalize_advantages_zero_mean_unit_std():
    adv = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    norm = _normalize_advantages(adv.copy(), eps=1e-8)
    flat = norm.reshape(-1)
    assert abs(float(flat.mean())) < 1e-6
    # Allow tiny tolerance due to float
    assert abs(float(flat.std()) - 1.0) < 1e-5


@pytest.mark.unit
def test_running_stats_updates_mean_std():
    rs = RunningStats()
    # Initial
    assert rs.mean() == 0.0 and rs.std() == 0.0
    rs.update(np.array([], dtype=np.float32))  # no-op
    rs.update(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    rs.update(np.array([4.0, -1.0, 0.0], dtype=np.float32))
    data = np.array([1.0, 2.0, 3.0, 4.0, -1.0, 0.0], dtype=np.float32)
    assert rs.count == data.size
    np.testing.assert_allclose(rs.mean(), float(data.mean()), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rs.std(), float(data.std()), rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_rolling_window_append_and_mean():
    with pytest.raises(ValueError):
        RollingWindow(0)
    rw = RollingWindow(3)
    assert rw.mean() == 0.0 and len(rw) == 0 and (not bool(rw))
    for v in [1.0, 2.0, 3.0]:
        rw.append(v)
    assert len(rw) == 3
    assert abs(rw.mean() - 2.0) < 1e-6
    # Appending beyond maxlen evicts oldest
    rw.append(4.0)
    assert len(rw) == 3
    assert abs(rw.mean() - ((2.0 + 3.0 + 4.0) / 3.0)) < 1e-6
    assert bool(rw)


class AlwaysNotDoneVecEnv:
    def __init__(self, num_envs=1, obs_dim=1):
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
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.array([False] * self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        self._obs = next_obs
        return next_obs.copy(), rewards, dones, infos


class CyclicPolicy(torch.nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self._step = 0
        self.dummy = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def device(self):
        return self.dummy.device

    def act(self, obs, deterministic=False):
        b = obs.shape[0]
        a_val = self._step % self.k
        self._step += 1
        a = torch.full((b,), a_val, dtype=torch.int64, device=obs.device)
        logp = torch.zeros(b, dtype=torch.float32, device=obs.device)
        v = torch.zeros(b, dtype=torch.float32, device=obs.device)
        return a, logp, v

    def predict_values(self, obs):
        return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)


@pytest.mark.unit
def test_action_histogram_counts_and_metrics():
    env = AlwaysNotDoneVecEnv(num_envs=1)
    policy = CyclicPolicy(k=3)  # actions sequence: 0,1,2,0,1
    collector = RolloutCollector(env, policy, n_steps=5, use_gae=True)
    _ = collector.collect()

    counts = collector.get_action_histogram_counts(reset=False)
    np.testing.assert_array_equal(counts, np.array([2, 2, 1], dtype=np.int64))

    # Metrics match histogram-derived mean/std
    m = collector.get_metrics()
    total = counts.sum()
    idxs = np.arange(counts.shape[0], dtype=np.float32)
    expected_mean = float((idxs * counts).sum() / total)
    var = float(((idxs - expected_mean) ** 2 * counts).sum() / total)
    expected_std = float(np.sqrt(max(0.0, var)))
    assert abs(m["policy/action_mean"] - expected_mean) < 1e-6
    assert abs(m["policy/action_std"] - expected_std) < 1e-6

    # Reset zeros internal counters
    counts2 = collector.get_action_histogram_counts(reset=True)
    np.testing.assert_array_equal(counts2, np.array([2, 2, 1], dtype=np.int64))
    counts3 = collector.get_action_histogram_counts(reset=False)
    np.testing.assert_array_equal(counts3, np.array([0, 0, 0], dtype=np.int64))


class SingleEnvWithTrailingPartial:
    """VecEnv with 1 env, terminal at t=1 within n_steps=4; then trailing partial."""

    def __init__(self):
        self.num_envs = 1
        self._step = 0
        self._obs = np.array([[0.0]], dtype=np.float32)

    def reset(self):
        self._step = 0
        self._obs[:] = 0.0
        return self._obs.copy()

    def step(self, actions):
        rewards_seq = [1.0, 2.0, 3.0, 4.0]
        dones_seq = [False, True, False, False]
        idx = min(self._step, len(rewards_seq) - 1)
        reward = np.array([rewards_seq[idx]], dtype=np.float32)
        done = np.array([dones_seq[idx]], dtype=bool)
        info = {"episode": {"r": float(sum(rewards_seq[: idx + 1])), "l": idx + 1}} if done[0] else {}
        infos = [info]
        next_obs = self._obs
        self._obs = next_obs
        self._step += 1
        return next_obs.copy(), reward, done, infos


class ZeroPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def device(self):
        return self.dummy.device

    def act(self, obs, deterministic=False):
        b = obs.shape[0]
        a = torch.zeros(b, dtype=torch.int64, device=obs.device)
        logp = torch.zeros(b, dtype=torch.float32, device=obs.device)
        v = torch.zeros(b, dtype=torch.float32, device=obs.device)
        return a, logp, v

    def predict_values(self, obs):
        return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)


@pytest.mark.unit
def test_slice_trajectories_remaps_trailing_partial_for_mc():
    env = SingleEnvWithTrailingPartial()
    policy = ZeroPolicy()
    collector = RolloutCollector(env, policy, n_steps=4, use_gae=False, normalize_advantages=False, gamma=1.0)
    traj = collector.collect()
    flat_returns = traj.returns.reshape(-1)
    # Sanity: with gamma=1 and terminal at t=1, returns = [3,2,7,4]
    np.testing.assert_array_equal(flat_returns.cpu().numpy(), np.array([3.0, 2.0, 7.0, 4.0], dtype=np.float32))

    # Remap indices 2 and 3 back to nearest previous valid index (1)
    out = collector.slice_trajectories(traj, np.array([2, 3]))
    np.testing.assert_array_equal(out.returns.cpu().numpy(), flat_returns.cpu().numpy()[[1, 1]])


@pytest.mark.unit
def test_collector_scalar_like_obs_shapes():
    class ScalarObsEnv:
        def __init__(self, num_envs=2):
            self.num_envs = num_envs
            # Return scalar-like observations as (n_envs, 1)
            self._obs = np.zeros((num_envs, 1), dtype=np.float32)
            self._step = 0

        def reset(self):
            self._step = 0
            self._obs.fill(0.0)
            return self._obs.copy()  # shape (n_envs, 1)

        def step(self, actions):
            self._step += 1
            next_obs = self._obs + 1.0
            rewards = np.ones(self.num_envs, dtype=np.float32)
            dones = np.array([False] * self.num_envs, dtype=bool)
            infos = [{} for _ in range(self.num_envs)]
            self._obs = next_obs
            return next_obs.copy(), rewards, dones, infos

    env = ScalarObsEnv(num_envs=2)
    policy = ZeroPolicy()
    collector = RolloutCollector(env, policy, n_steps=3, use_gae=True)
    traj = collector.collect()
    # Observations/next_observations should be (n_envs*T, 1)
    assert traj.observations.shape == (env.num_envs * 3, 1)
    assert traj.next_observations.shape == (env.num_envs * 3, 1)
