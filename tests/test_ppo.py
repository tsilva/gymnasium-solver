import math
from types import SimpleNamespace

import pytest
import torch

from agents.ppo import PPO


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
    agent = object.__new__(PPO)
    agent.config = SimpleNamespace(
        normalize_advantages="off",  # do not renormalize in test
        ent_coef=0.0,
        vf_coef=0.0,
        policy_lr=3e-4,
        hidden_dims=(64, 64),
    )
    agent.clip_range = clip
    agent.policy_model = _FakePolicy(new_logps, values)
    agent.log_metrics = lambda *a, **k: None  # type: ignore

    batch = SimpleNamespace(
        observations=states,
        actions=actions,
        log_prob=old_logps,
        advantages=advantages,
        returns=returns,
    )

    loss = agent.losses_for_batch(batch, batch_idx=0)

    # Compute expected clipped policy loss manually
    unclipped = advantages * ratios
    clipped_ratios = torch.clamp(ratios, 1.0 - clip, 1.0 + clip)
    clipped = advantages * clipped_ratios
    expected_policy_loss = -torch.min(unclipped, clipped).mean()

    assert torch.isclose(loss, expected_policy_loss, atol=1e-6)


@pytest.mark.unit
def test_ppo_clip_range_schedule_update():
    initial = 0.3
    progress = 0.75  # 75% of training -> clip should be 25% of initial

    agent = object.__new__(PPO)
    agent.config = SimpleNamespace(clip_range=initial, clip_range_schedule="linear")
    agent.clip_range = initial

    seen = {}

    def _log(m, prefix=None):  # noqa: ARG001
        seen.update(m)

    agent.log_metrics = _log  # type: ignore
    agent._get_training_progress = lambda: progress  # type: ignore

    agent._update_schedules__clip_range()

    expected = max(initial * (1.0 - progress), 0.0)
    assert math.isclose(agent.clip_range, expected, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(seen.get("clip_range", -1.0), expected, rel_tol=0, abs_tol=1e-12)


@pytest.mark.unit
def test_ppo_create_models_and_optimizer():
    # Minimal train_env stub with required API
    class _Env:
        def get_input_dim(self):
            return 4

        def get_output_dim(self):
            return 3

    agent = object.__new__(PPO)
    agent.config = SimpleNamespace(hidden_dims=(32, 32), policy_lr=1e-3)
    agent.train_env = _Env()

    # Create model
    agent.create_models()
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
