from pathlib import Path

import pytest

from utils.run import Run


class DummyRun:
    def __init__(self, id):
        self.id = id


@pytest.mark.unit
def test_run_manager_creates_dirs_and_symlink(tmp_path: Path, monkeypatch):
    from utils.config import Config

    # Simulate a run and initialize via Run API
    monkeypatch.chdir(tmp_path)

    config = Config(
        env_id="CartPole-v1",
        algo_id="ppo",
        seed=42,
    )

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
