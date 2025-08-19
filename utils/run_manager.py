"""Run directory management utilities for organizing all run assets."""

from pathlib import Path

class RunManager:
    """Manages run-specific directories and assets organization."""

    def __init__(self, run_id :str, base_runs_dir: str = "runs"):
        # Ensure run dir exists
        self._base_runs_dir = Path(base_runs_dir)
        self._run_id = run_id
        self._run_dir = self._base_runs_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Make `latest-run` symlink to this run
        latest = self._base_runs_dir / "latest-run"
        if latest.exists() or latest.is_symlink(): latest.unlink()
        latest.symlink_to(Path(self._run_id))
    
    def ensure_path(self, path: str | Path) -> Path:
        from pathlib import Path
        full_path = self._run_dir / Path(path)
        target_dir = full_path.parent if full_path.suffix else full_path
        target_dir.mkdir(parents=True, exist_ok=True)
        return full_path
    
    def get_run_dir(self) -> Path:
        """Returns the directory for the current run."""
        return self._run_dir
    
    def get_run_id(self) -> str:
        """Returns the ID of the current run."""
        return self._run_id