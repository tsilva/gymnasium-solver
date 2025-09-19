from pathlib import Path

import pytest

from utils.run import Run


class DummyRun:
    def __init__(self, id):
        self.id = id


@pytest.mark.unit
def test_run_manager_creates_dirs_and_symlink(tmp_path: Path, monkeypatch):
    # Simulate a wandb run and initialize via Run API
    run = DummyRun("abc123")
    run_obj = Run.from_wandb_run(wandb_run=run, runs_root=tmp_path / "runs")
    run_dir = run_obj.get_run_dir()
    assert run_dir.exists()
    assert (run_dir / "checkpoints").exists()
    # 'videos' directory is created lazily by video components; not at run setup
    assert not (run_dir / "videos").exists()
    # configs subdir is optional in new layout

    latest = Path(tmp_path / "runs" / "@latest-run")
    assert latest.is_symlink()
    assert latest.readlink() == Path("abc123")

    # Save a config file
    cfg_path = run_obj.save_config({"a": 1}, filename="cfg.json")
    assert cfg_path.exists()
