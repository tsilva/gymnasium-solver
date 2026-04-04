from pathlib import Path

import pytest

from utils.run import Run, resolve_checkpoint_dir


@pytest.mark.unit
def test_checkpoint_path_prefers_model_pt_over_legacy_policy_ckpt(tmp_path: Path, monkeypatch):
    from utils.config import load_config

    monkeypatch.chdir(tmp_path)
    run = Run.create(run_id="abc123", config=load_config("Bandit-v0", "ppo"))
    epoch_dir = run.checkpoint_dir_for_epoch(3)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    (epoch_dir / "model.pt").write_text("new")
    (epoch_dir / "policy.ckpt").write_text("old")
    run.best_checkpoint_dir.symlink_to(epoch_dir.resolve(), target_is_directory=True)

    assert run.best_checkpoint_path.resolve() == (epoch_dir / "model.pt").resolve()


@pytest.mark.unit
def test_checkpoint_path_falls_back_to_legacy_policy_ckpt(tmp_path: Path, monkeypatch):
    from utils.config import load_config

    monkeypatch.chdir(tmp_path)
    run = Run.create(run_id="abc123", config=load_config("Bandit-v0", "ppo"))
    epoch_dir = run.checkpoint_dir_for_epoch(4)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    (epoch_dir / "policy.ckpt").write_text("legacy")
    run.last_checkpoint_dir.symlink_to(epoch_dir.resolve(), target_is_directory=True)

    assert run.last_checkpoint_path.resolve() == (epoch_dir / "policy.ckpt").resolve()


@pytest.mark.unit
def test_resolve_checkpoint_dir_uses_last_when_best_is_missing(tmp_path: Path, monkeypatch):
    from utils.config import load_config

    monkeypatch.chdir(tmp_path)
    run = Run.create(run_id="abc123", config=load_config("Bandit-v0", "ppo"))
    epoch_dir = run.checkpoint_dir_for_epoch(5)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    run.last_checkpoint_dir.symlink_to(epoch_dir.resolve(), target_is_directory=True)

    assert resolve_checkpoint_dir(run, None) == run.last_checkpoint_dir
