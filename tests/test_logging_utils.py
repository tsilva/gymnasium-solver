import sys
from pathlib import Path

import pytest

from utils.logging import LogFileManager, capture_all_output, log_config_details


class DummyCfg:
    algo_id = "ppo"
    env_id = "CartPole-v1"
    seed = 123


@pytest.mark.unit
def test_logfile_manager_creates_file(tmp_path: Path):
    lm = LogFileManager(log_dir=str(tmp_path / "logs"), max_log_files=3)
    f = lm.create_log_file(DummyCfg())
    try:
        assert f.writable()
        p = Path(f.name)
        assert p.exists()
        assert "training_" in p.name
    finally:
        f.close()


@pytest.mark.unit
def test_capture_all_output_writes_to_file(tmp_path: Path):
    log_dir = tmp_path / "logs"
    with capture_all_output(DummyCfg(), log_dir=str(log_dir)) as f:
        print("hello world")
        sys.stderr.write("error line\n")
        log_config_details(DummyCfg(), f)
    # After context, file should be closed and content present
    files = list(log_dir.glob("training_*.log"))
    assert files, "no log file created"
    text = files[0].read_text()
    assert "hello world" in text
    assert "error line" in text
    assert "Configuration Details" in text
