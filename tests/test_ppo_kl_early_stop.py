from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from agents.ppo.ppo_agent import PPOAgent
from utils.config import PPOConfig
from utils.run_manager import RunManager


def _make_config(*, target_kl: float | None) -> PPOConfig:
    return PPOConfig(
        env_id="CartPole-v1",
        n_envs=1,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        max_timesteps=None,
        target_kl=target_kl,
        eval_episodes=1,
        eval_warmup_epochs=0,
    )


def test_ppo_config_requires_positive_target_kl():
    with pytest.raises(ValueError):
        _make_config(target_kl=0.0)


class _DummyDist:
    def __init__(self, log_prob: float, batch_size: int, dtype: torch.dtype, device: torch.device):
        self._log_prob = log_prob
        self._batch_size = batch_size
        self._dtype = dtype
        self._device = device

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.full((self._batch_size,), self._log_prob, dtype=self._dtype, device=self._device)

    def entropy(self) -> torch.Tensor:
        return torch.zeros(self._batch_size, dtype=self._dtype, device=self._device)


class _DummyPolicy(nn.Module):
    def __init__(self, log_prob: float):
        super().__init__()
        self._log_prob = log_prob

    def forward(self, states: torch.Tensor):
        batch_size = states.shape[0]
        dist = _DummyDist(self._log_prob, batch_size, states.dtype, states.device)
        values = torch.zeros(batch_size, dtype=states.dtype, device=states.device)
        return dist, values

    def compute_grad_norms(self):
        return {}


def test_ppo_kl_threshold_skips_remaining_batches(tmp_path):
    config = _make_config(target_kl=0.01)
    agent = PPOAgent(config)
    agent.run_manager = RunManager(run_id="test-run", base_runs_dir=str(tmp_path))

    # Avoid touching real optimizers / parameters in the test
    agent._backpropagate_and_step = lambda *_, **__: None
    agent.policy_model = _DummyPolicy(log_prob=0.2)

    batch = SimpleNamespace(
        observations=torch.zeros(4, 3),
        actions=torch.zeros(4, dtype=torch.float32),
        logprobs=torch.zeros(4, dtype=torch.float32),
        advantages=torch.ones(4, dtype=torch.float32),
        returns=torch.ones(4, dtype=torch.float32),
    )

    try:
        agent.training_step(batch, 0)
        assert agent._kl_stop_triggered is True

        def fail(*args, **kwargs):
            raise AssertionError("losses_for_batch should not run after KL stop")

        agent.losses_for_batch = fail  # type: ignore[assignment]
        agent.training_step(batch, 1)
    finally:
        for stage in ("train", "val", "test"):
            agent.get_env(stage).close()
