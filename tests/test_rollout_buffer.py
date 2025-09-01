import numpy as np
import pytest
import torch

from utils.rollouts import RolloutBuffer


@pytest.mark.unit
def test_begin_rollout_wrap_and_size_tracking():
    buf = RolloutBuffer(n_envs=2, obs_shape=(3,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=5)

    s0 = buf.begin_rollout(3)
    assert s0 == 0 and buf.pos == 3 and buf.size == 3

    s1 = buf.begin_rollout(2)
    assert s1 == 3 and buf.pos == 5 and buf.size == 5

    # This requires wrap-around; size should remain the max ever written
    s2 = buf.begin_rollout(2)
    assert s2 == 0 and buf.pos == 2 and buf.size == 5


@pytest.mark.unit
def test_begin_rollout_exceeds_maxsize_raises():
    buf = RolloutBuffer(n_envs=1, obs_shape=(1,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=4)
    with pytest.raises(ValueError):
        buf.begin_rollout(5)


@pytest.mark.unit
def test_add_stores_observations_and_dtypes_correctly():
    # Provide correctly-shaped obs for obs_shape (2,2)
    buf = RolloutBuffer(n_envs=2, obs_shape=(2, 2), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=3)
    idx = buf.begin_rollout(1)

    # Build distinct per-env values to verify assignment
    obs_np = np.array([
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
    ], dtype=np.float32)
    next_obs_np = obs_np + 1
    actions_np = np.array([1, 2], dtype=np.int64)
    logps_np = np.array([0.1, 0.2], dtype=np.float32)
    values_np = np.array([0.3, 0.4], dtype=np.float32)
    rewards_np = np.array([0.0, 0.0], dtype=np.float32)
    dones_np = np.array([False, False], dtype=bool)
    timeouts_np = np.array([False, False], dtype=bool)

    buf.add(idx, obs_np, next_obs_np, actions_np, logps_np, values_np, rewards_np, dones_np, timeouts_np)

    # Observations should match exactly what we provided
    np.testing.assert_array_equal(buf.obs_buf[idx, 0], np.array([[0, 1], [2, 3]], dtype=np.float32))
    np.testing.assert_array_equal(buf.obs_buf[idx, 1], np.array([[4, 5], [6, 7]], dtype=np.float32))

    # Dtypes in CPU buffers
    assert buf.actions_buf.dtype == np.int64
    assert buf.logprobs_buf.dtype == np.float32
    assert buf.values_buf.dtype == np.float32


@pytest.mark.unit
def test_add_bad_obs_shape_raises():
    buf = RolloutBuffer(n_envs=2, obs_shape=(3,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=2)
    idx = buf.begin_rollout(1)
    # Provide obs with wrong shape (expected (2,3))
    bad_obs = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(AssertionError):
        buf.add(
            idx,
            bad_obs,
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            np.zeros(2, dtype=bool),
        )


@pytest.mark.unit
def test_flatten_slice_env_major_order_and_shapes_vector_obs():
    n_envs, T, D = 3, 4, 5
    buf = RolloutBuffer(n_envs=n_envs, obs_shape=(D,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=T)
    start = buf.begin_rollout(T)

    # Fill with deterministic values to check flattening order
    for t in range(T):
        # observations per env are all equal to (env_id + 10*t) for the first feature
        obs = np.stack([np.full(D, env + 10 * t, dtype=np.float32) for env in range(n_envs)])
        next_obs = obs + 1
        actions = np.array([env + 100 * t for env in range(n_envs)], dtype=np.int64)
        rewards = np.array([t + 0.5 * env for env in range(n_envs)], dtype=np.float32)
        dones = np.array([False] * n_envs, dtype=bool)
        timeouts = np.array([False] * n_envs, dtype=bool)

        logps = np.zeros(n_envs, dtype=np.float32)
        values = np.zeros(n_envs, dtype=np.float32)

        buf.add(start + t, obs, next_obs, actions, logps, values, rewards, dones, timeouts)

    adv = np.zeros((T, n_envs), dtype=np.float32)
    ret = np.zeros((T, n_envs), dtype=np.float32)
    traj = buf.flatten_slice_env_major(start, start + T, adv, ret)

    # Shapes
    assert traj.observations.shape == (n_envs * T, D)
    assert traj.next_observations.shape == (n_envs * T, D)
    assert traj.actions.shape == (n_envs * T,)
    assert traj.rewards.shape == (n_envs * T,)
    assert traj.dones.shape == (n_envs * T,)

    # Env-major order: all times for env0, then env1, etc., for actions
    expected_actions = []
    for env in range(n_envs):
        for t in range(T):
            expected_actions.append(env + 100 * t)
    np.testing.assert_array_equal(traj.actions.cpu().numpy(), np.array(expected_actions, dtype=np.int64))

    # First feature of observations follows the same env-major structure
    obs_first_feat = traj.observations[:, 0].cpu().numpy()
    expected_first_feat = []
    for env in range(n_envs):
        for t in range(T):
            expected_first_feat.append(env + 10 * t)
    np.testing.assert_array_equal(obs_first_feat, np.array(expected_first_feat, dtype=np.float32))


@pytest.mark.unit
def test_flatten_slice_scalar_obs_shapes_and_dtypes():
    # Exercise the branch where next_obs_buf.ndim == 2 (scalar observations)
    n_envs, T = 2, 3
    buf = RolloutBuffer(n_envs=n_envs, obs_shape=(), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=T)
    start = buf.begin_rollout(T)

    for t in range(T):
        obs = np.array([t, t + 1], dtype=np.float32)  # shape (n_envs,)
        next_obs = obs + 0.5
        actions = np.array([t, t + 10], dtype=np.int64)
        rewards = np.array([1.0, 2.0], dtype=np.float32)
        dones = np.array([False, True] if t == T - 1 else [False, False], dtype=bool)
        timeouts = np.array([False, False], dtype=bool)

        logps = np.zeros(n_envs, dtype=np.float32)
        values = np.zeros(n_envs, dtype=np.float32)

        buf.add(start + t, obs, next_obs, actions, logps, values, rewards, dones, timeouts)

    adv = np.zeros((T, n_envs), dtype=np.float32)
    ret = np.zeros((T, n_envs), dtype=np.float32)
    traj = buf.flatten_slice_env_major(start, start + T, adv, ret)

    # Observations should be (n_envs*T, 1) and next_observations should match shape
    assert traj.observations.shape == (n_envs * T, 1)
    assert traj.next_observations.shape == (n_envs * T, 1)

    # Dtypes
    assert traj.actions.dtype == torch.int64
    assert traj.rewards.dtype == torch.float32
    assert traj.dones.dtype == torch.bool
    assert traj.advantages.dtype == torch.float32
    assert traj.returns.dtype == torch.float32


@pytest.mark.unit
def test_store_cpu_step_and_flatten_propagates_dones_and_rewards():
    n_envs, T = 2, 2
    buf = RolloutBuffer(n_envs=n_envs, obs_shape=(4,), obs_dtype=np.float32, device=torch.device("cpu"), maxsize=T)
    start = buf.begin_rollout(T)

    # Step 0: no dones
    buf.add(
        start + 0,
        obs_np=np.zeros((n_envs, 4), dtype=np.float32),
        next_obs_np=np.ones((n_envs, 4), dtype=np.float32),
        actions_np=np.array([1, 2], dtype=np.int64),
        logps_np=np.zeros(n_envs, dtype=np.float32),
        values_np=np.zeros(n_envs, dtype=np.float32),
        rewards_np=np.array([0.5, 1.5], dtype=np.float32),
        dones_np=np.array([False, False], dtype=bool),
        timeouts_np=np.array([False, False], dtype=bool),
    )

    # Step 1: env 1 done
    buf.add(
        start + 1,
        obs_np=np.ones((n_envs, 4), dtype=np.float32),
        next_obs_np=2 * np.ones((n_envs, 4), dtype=np.float32),
        actions_np=np.array([3, 4], dtype=np.int64),
        logps_np=np.zeros(n_envs, dtype=np.float32),
        values_np=np.zeros(n_envs, dtype=np.float32),
        rewards_np=np.array([2.5, 3.5], dtype=np.float32),
        dones_np=np.array([False, True], dtype=bool),
        timeouts_np=np.array([False, False], dtype=bool),
    )

    adv = np.zeros((T, n_envs), dtype=np.float32)
    ret = np.zeros((T, n_envs), dtype=np.float32)
    traj = buf.flatten_slice_env_major(start, start + T, adv, ret)

    # Rewards in env-major order: env0 [0.5, 2.5], env1 [1.5, 3.5]
    np.testing.assert_array_equal(traj.rewards.cpu().numpy(), np.array([0.5, 2.5, 1.5, 3.5], dtype=np.float32))
    # Dones in env-major order: env0 [F, F], env1 [F, T]
    np.testing.assert_array_equal(traj.dones.cpu().numpy(), np.array([False, False, False, True]))
