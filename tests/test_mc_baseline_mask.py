import gymnasium as gym
import numpy as np
import torch

from utils.rollout_collector import RolloutCollector


class SingleEnvWithTrailingPartial:
    """VecEnv with 1 env, terminal at t=1 within n_steps=4; then trailing partial.

    Rewards: [1.0, 2.0, 3.0, 4.0]
    Dones:   [F,   T,   F,   F]
    """

    def __init__(self):
        self.num_envs = 1
        self._step = 0
        self._obs = np.array([[0.0]], dtype=np.float32)
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space
        self.single_action_space = gym.spaces.Discrete(2)
        self.action_space = self.single_action_space

    def reset(self):
        self._step = 0
        self._obs[:] = 0.0
        return self._obs.copy(), {}

    def step(self, actions):
        # Define rewards and dones per step
        rewards_seq = [1.0, 2.0, 3.0, 4.0]
        dones_seq = [False, True, False, False]
        idx = min(self._step, len(rewards_seq) - 1)
        reward = np.array([rewards_seq[idx]], dtype=np.float32)
        terminated = np.array([dones_seq[idx]], dtype=bool)
        truncated = np.array([False], dtype=bool)
        infos = {}
        if terminated[0]:
            infos = {
                "episode": {
                    "r": np.array([float(sum(rewards_seq[: idx + 1]))], dtype=np.float32),
                    "l": np.array([idx + 1], dtype=np.int32),
                },
                "_episode": terminated.copy(),
            }

        # Next obs stays constant shape
        next_obs = self._obs
        self._obs = next_obs
        self._step += 1
        return next_obs.copy(), reward, terminated, truncated, infos


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def device(self):
        return self.dummy.device

    def forward(self, obs):
        b = obs.shape[0]
        logits = torch.zeros((b, 2), dtype=torch.float32, device=obs.device)
        logits[:, 0] = 10.0
        dist = torch.distributions.Categorical(logits=logits)
        value = torch.zeros(b, dtype=torch.float32, device=obs.device)
        return dist, value


def test_mc_baseline_uses_masked_values_only():
    env = SingleEnvWithTrailingPartial()
    policy = DummyPolicy()
    collector = RolloutCollector(
        env,
        policy,
        n_steps=4,
        use_gae=False,              # Monte Carlo
        normalize_advantages=False,
        gamma=1.0,                  # simplify returns to sums ahead
        returns_type="reward_to_go",
    )

    _ = collector.collect()
    metrics = collector.get_metrics()

    # Valid positions are up to and including the last terminal (t=1)
    # Returns with gamma=1: t=0 -> 1+2=3, t=1 -> 2
    expected_mean = (3.0 + 2.0) / 2.0
    assert abs(metrics["roll/baseline/mean"] - expected_mean) < 1e-6
