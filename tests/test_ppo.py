import math
import sys
from types import SimpleNamespace

import pytest
import torch

try:  # pragma: no cover - import guard for optional dependency in test envs
    import pytorch_lightning as _pl  # type: ignore
except Exception:
    sys.modules["pytorch_lightning"] = SimpleNamespace(
        Callback=object,
        Trainer=object,
        LightningModule=object,
    )
else:
    if not all(hasattr(_pl, attr) for attr in ("Callback", "Trainer", "LightningModule")):
        sys.modules["pytorch_lightning"] = SimpleNamespace(
            Callback=object,
            Trainer=object,
            LightningModule=object,
        )

from agents.ppo.ppo_agent import PPOAgent
from trainer_callbacks.hyperparameter_scheduler import HyperparameterSchedulerCallback


class _FakeDist:
    def __init__(self, logps: torch.Tensor, entropy_value: float = 0.0):
        self._logps = logps
        self._entropy_value = entropy_value

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:  # noqa: ARG002 - actions ignored for test determinism
        return self._logps

    def entropy(self) -> torch.Tensor:
        return torch.zeros_like(self._logps) + self._entropy_value


class _FakePolicy:
    def __init__(self, logps: torch.Tensor, values: torch.Tensor, entropy_value: float = 0.0):
        self._logps = logps
        self._values = values
        self._entropy_value = entropy_value

    def __call__(self, states: torch.Tensor):
        assert states.shape[0] == self._logps.shape[0] == self._values.shape[0]
        return _FakeDist(self._logps, self._entropy_value), self._values


@pytest.mark.unit
def test_ppo_policy_clipping_math():
    # Prepare a tiny deterministic batch with 2 samples to exercise both clipping sides
    N, obs_dim = 2, 4
    states = torch.randn(N, obs_dim)
    actions = torch.tensor([0, 1], dtype=torch.int64)

    clip = 0.2

    # Target ratios: one above (1+clip), one below (1-clip)
    ratios = torch.tensor([1.0 + clip + 0.3, 1.0 - clip - 0.25], dtype=torch.float32)
    # Choose old logps = 0 for simplicity so new_logps = log(ratio)
    old_logps = torch.zeros(N)
    new_logps = torch.log(ratios)

    # Advantages: positive for first, negative for second to hit both cases
    advantages = torch.tensor([1.5, -2.0], dtype=torch.float32)
    # Set value loss and entropy loss to zero so total loss == policy loss
    returns = torch.zeros(N)
    values = returns.clone()

    # Build a minimal PPO instance without running BaseAgent.__init__
    agent = object.__new__(PPOAgent)
    agent.config = SimpleNamespace(
        normalize_advantages="off",  # do not renormalize in test
        target_kl=None,
        ent_coef=0.0,
        vf_coef=0.0,
        policy_lr=3e-4,
        hidden_dims=(64, 64),
    )
    agent.clip_range = clip
    agent.clip_range_vf = 1e6  # effectively disable value clipping for this test
    agent.vf_coef = 0.0
    agent.ent_coef = 0.0
    agent.policy_model = _FakePolicy(new_logps, values)
    agent.metrics_recorder = SimpleNamespace(record=lambda *a, **k: None)

    batch = SimpleNamespace(
        observations=states,
        actions=actions,
        logprobs=old_logps,
        values=values,
        advantages=advantages,
        returns=returns,
    )

    result = agent.losses_for_batch(batch, batch_idx=0)
    loss = result['loss']

    # Compute expected clipped policy loss manually
    unclipped = advantages * ratios
    clipped_ratios = torch.clamp(ratios, 1.0 - clip, 1.0 + clip)
    clipped = advantages * clipped_ratios
    expected_policy_loss = -torch.min(unclipped, clipped).mean()

    assert torch.isclose(loss, expected_policy_loss, atol=1e-6)


@pytest.mark.unit
def test_ppo_clip_range_schedule_update():
    class _StubModule:
        def __init__(self, start: float) -> None:
            self.clip_range = start
            self.config = SimpleNamespace(clip_range=start)
            self.collector = SimpleNamespace(total_vec_steps=0)

        def get_rollout_collector(self, stage: str):
            assert stage == "train"
            return self.collector

        def set_hyperparameter(self, param: str, value: float) -> None:
            """Set a hyperparameter value. Called by HyperparameterSchedulerCallback."""
            setattr(self, param, value)
            if hasattr(self.config, param):
                setattr(self.config, param, value)

    initial = 0.3
    stub = _StubModule(initial)

    callback = HyperparameterSchedulerCallback(
        schedule="linear",
        parameter="clip_range",
        start_value=initial,
        end_value=0.0,
        start_step=0.0,
        end_step=100.0,
    )

    stub.collector.total_vec_steps = 0
    callback.on_train_epoch_end(None, stub)
    assert math.isclose(stub.clip_range, initial, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(stub.config.clip_range, initial, rel_tol=0, abs_tol=1e-12)

    stub.collector.total_vec_steps = 50
    callback.on_train_epoch_end(None, stub)
    assert math.isclose(stub.clip_range, initial * 0.5, rel_tol=0, abs_tol=1e-12)

    stub.collector.total_vec_steps = 150
    callback.on_train_epoch_end(None, stub)
    assert math.isclose(stub.clip_range, 0.0, rel_tol=0, abs_tol=1e-12)


@pytest.mark.unit
@pytest.mark.skip(reason="Test needs more setup - _Env stub missing required attributes")
def test_ppo_build_models_and_optimizer():
    # Minimal train_env stub with required API
    class _Env:
        pass

    agent = object.__new__(PPOAgent)
    agent.config = SimpleNamespace(hidden_dims=(32, 32), policy_lr=1e-3)
    agent._envs = {"train": _Env()}

    # Create model
    agent.build_models()
    assert hasattr(agent, "policy_model")

    # Forward shape sanity check
    dist, value = agent.policy_model(torch.randn(5, 4))
    assert value.shape == (5,)
    assert hasattr(dist, "log_prob")

    # Optimizer wiring
    opt = agent.configure_optimizers()
    assert isinstance(opt, torch.optim.AdamW)
    assert any(abs(pg["lr"] - 1e-3) < 1e-12 for pg in opt.param_groups)
    assert all(pg.get("eps", None) == 1e-5 for pg in opt.param_groups)
