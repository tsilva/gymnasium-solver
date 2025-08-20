import types
from pathlib import Path

import pytest
import torch

from utils.checkpoint import (
    find_latest_checkpoint,
    list_available_checkpoints,
    load_checkpoint,
)


class DummyAgent:
    def __init__(self):
        self.policy_model = types.SimpleNamespace(load_state_dict=lambda s: None)
        opt = types.SimpleNamespace(load_state_dict=lambda s: None)
        self.optimizers = lambda: opt
        self.total_timesteps = 0
        self.best_eval_reward = float('-inf')


@pytest.mark.unit
def test_find_and_list_checkpoints(tmp_path: Path):
    base = tmp_path / "checkpoints" / "ppo" / "CartPole-v1"
    base.mkdir(parents=True)

    # Create multiple files with different names
    (base / "epoch=3-step=300.ckpt").write_text("x")
    (base / "best.ckpt").write_text("x")
    (base / "last.ckpt").write_text("x")
    (base / "threshold-epoch=2-step=200.ckpt").write_text("x")

    latest = find_latest_checkpoint("ppo", "CartPole-v1", checkpoint_dir=str(tmp_path / "checkpoints"))
    # Preference order chooses best then last
    assert latest.name in {"best.ckpt", "last.ckpt", "best.ckpt", "last.ckpt"}

    listing = list_available_checkpoints(checkpoint_dir=str(tmp_path / "checkpoints"))
    assert "ppo" in listing and "CartPole-v1" in listing["ppo"]
    assert any(name.endswith(".ckpt") for name in listing["ppo"]["CartPole-v1"])  # has entries


@pytest.mark.unit
def test_load_checkpoint_roundtrip(tmp_path: Path):
    # Prepare a fake checkpoint file
    ckpt_dir = tmp_path
    ckpt = ckpt_dir / "ck.ckpt"
    torch.save({
        "model_state_dict": {"a": torch.tensor([1.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "epoch": 5,
        "total_timesteps": 1234,
        "best_eval_reward": 10.0,
        "current_eval_reward": 9.0,
        "is_best": True,
        "is_threshold": False,
        "rng_states": {"torch": torch.get_rng_state(), "torch_cuda": None},
    }, ckpt)

    agent = DummyAgent()
    out = load_checkpoint(ckpt, agent, resume_training=True)
    assert out["epoch"] == 5
    assert agent.total_timesteps == 1234
    assert agent.best_eval_reward == pytest.approx(10.0)
