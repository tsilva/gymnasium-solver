from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from agents.ppo.ppo_agent import PPOAgent
from utils.config import PPOConfig
def _make_config(*, target_kl: float | None) -> PPOConfig:
    return PPOConfig(
        env_id="CartPole-v1",
        model_id="mlp_tiny",
        n_envs=1,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
        target_kl=target_kl,
        eval_episodes=1,
        eval_warmup_epochs=0,
        enable_wandb=False,
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

    def compute_activation_stats(self):
        return {}


def test_ppo_kl_threshold_skips_remaining_batches(tmp_path):
    config = _make_config(target_kl=0.01)
    agent = PPOAgent(config)
    agent.run = SimpleNamespace()

    # Avoid touching real optimizers / parameters in the test
    agent._backpropagate_and_step = lambda *_, **__: None
    agent.policy_model = _DummyPolicy(log_prob=0.2)

    batch = SimpleNamespace(
        observations=torch.zeros(4, 3),
        actions=torch.zeros(4, dtype=torch.float32),
        logprobs=torch.zeros(4, dtype=torch.float32),
        values=torch.zeros(4, dtype=torch.float32),
        advantages=torch.ones(4, dtype=torch.float32),
        returns=torch.arange(1, 5, dtype=torch.float32),
    )

    try:
        agent.training_step(batch, 0)
        assert agent._early_stop_epoch is True

        def fail(*args, **kwargs):
            raise AssertionError("losses_for_batch should not run after KL stop")

        agent.losses_for_batch = fail  # type: ignore[assignment]
        agent.training_step(batch, 1)
    finally:
        for stage in ("train", "val", "test"):
            agent.get_env(stage).close()


def test_ppo_kl_threshold_resets_on_next_epoch(tmp_path):
    config = _make_config(target_kl=0.01)
    agent = PPOAgent(config)
    agent.run = SimpleNamespace()
    agent._backpropagate_and_step = lambda *_, **__: None
    agent.policy_model = _DummyPolicy(log_prob=0.2)
    agent.current_epoch = 0
    agent._print_metrics_logger = None
    agent.timings = SimpleNamespace(start=lambda *_, **__: None)
    agent.trainer = SimpleNamespace(should_stop=False)
    agent._read_hyperparameters_from_run = lambda: None
    agent._log_hyperparameters = lambda: None

    class _Collector:
        def __init__(self):
            self.collect_calls = 0

        def get_metrics(self):
            return {"cnt/total_env_steps": 0}

        def collect(self):
            self.collect_calls += 1
            return SimpleNamespace()

    collector = _Collector()
    agent._rollout_collectors["train"] = collector

    batch = SimpleNamespace(
        observations=torch.zeros(4, 3),
        actions=torch.zeros(4, dtype=torch.float32),
        logprobs=torch.zeros(4, dtype=torch.float32),
        values=torch.zeros(4, dtype=torch.float32),
        advantages=torch.ones(4, dtype=torch.float32),
        returns=torch.arange(1, 5, dtype=torch.float32),
    )

    calls = {"count": 0}
    original_losses_for_batch = agent.losses_for_batch

    def counted_losses_for_batch(*args, **kwargs):
        calls["count"] += 1
        return original_losses_for_batch(*args, **kwargs)

    agent.losses_for_batch = counted_losses_for_batch  # type: ignore[assignment]

    try:
        agent.training_step(batch, 0)
        assert agent._early_stop_epoch is True
        assert calls["count"] == 1

        agent.current_epoch = 1
        agent.on_train_epoch_start()
        assert agent._early_stop_epoch is False
        assert collector.collect_calls == 1

        agent.training_step(batch, 0)
        assert calls["count"] == 2
    finally:
        for stage in ("train", "val", "test"):
            agent.get_env(stage).close()
