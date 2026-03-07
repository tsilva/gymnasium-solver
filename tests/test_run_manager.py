from pathlib import Path
import types
import sys

import pytest

from utils.run import Run, ensure_run_dir, resolve_checkpoint_dir, resolve_run_id


class DummyRun:
    def __init__(self, id):
        self.id = id


@pytest.mark.unit
def test_run_manager_creates_dirs_and_symlink(tmp_path: Path, monkeypatch):
    from utils.config import load_config

    # Simulate a run and initialize via Run API
    monkeypatch.chdir(tmp_path)

    config = load_config("Bandit-v0", "ppo")

    run_obj = Run.create(run_id="abc123", config=config)
    run_dir = Path(run_obj.run_dir)

    assert run_dir.exists()
    assert run_obj.config_path.exists()
    # Checkpoints directory is not created until checkpoints are saved
    assert not run_obj.checkpoints_dir.exists()
    # Videos are stored directly in checkpoint directories, no separate videos/ folder

    latest = Path("runs/@last")
    assert latest.is_symlink()
    assert latest.resolve() == run_dir.resolve()


@pytest.mark.unit
def test_resolve_run_id_expands_last_symlink(tmp_path: Path, monkeypatch):
    from utils.config import load_config

    monkeypatch.chdir(tmp_path)
    Run.create(
        run_id="abc123",
        config=load_config("Bandit-v0", "ppo"),
    )

    assert resolve_run_id("@last") == "abc123"
    assert resolve_run_id("explicit") == "explicit"


@pytest.mark.unit
def test_ensure_run_dir_downloads_missing_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fake_module = types.ModuleType("utils.wandb_artifacts")

    def fake_download_run_artifact(run_id: str) -> None:
        (tmp_path / "runs" / run_id).mkdir(parents=True, exist_ok=True)

    fake_module.download_run_artifact = fake_download_run_artifact
    monkeypatch.setitem(sys.modules, "utils.wandb_artifacts", fake_module)

    run_dir = ensure_run_dir("remote123")

    assert run_dir.resolve() == (tmp_path / "runs" / "remote123").resolve()
    assert run_dir.exists()


@pytest.mark.unit
def test_resolve_checkpoint_dir_supports_symlinks_and_epochs(tmp_path: Path, monkeypatch):
    from utils.config import load_config

    monkeypatch.chdir(tmp_path)
    run = Run.create(
        run_id="abc123",
        config=load_config("Bandit-v0", "ppo"),
    )

    epoch_dir = run.checkpoint_dir_for_epoch(3)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    run.last_checkpoint_dir.symlink_to(epoch_dir.resolve(), target_is_directory=True)
    run.best_checkpoint_dir.symlink_to(epoch_dir.resolve(), target_is_directory=True)

    assert resolve_checkpoint_dir(run, None) == run.best_checkpoint_dir
    assert resolve_checkpoint_dir(run, "@best") == run.best_checkpoint_dir
    assert resolve_checkpoint_dir(run, "@last") == run.last_checkpoint_dir
    assert resolve_checkpoint_dir(run, "3") == epoch_dir
    assert resolve_checkpoint_dir(run, "epoch=3") == epoch_dir
