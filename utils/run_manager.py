"""Run directory management utilities for organizing all run assets."""

from pathlib import Path

class RunManager:
    """Manages run-specific directories and assets organization."""

    def __init__(self, run_id: str | None = None, base_runs_dir: str = "runs"):
        # Only set up directories when a run id is provided
        self._base_runs_dir = Path(base_runs_dir)
        self._run_id = run_id or ""
        self._run_dir = self._base_runs_dir / self._run_id if self._run_id else self._base_runs_dir
        if self._run_id:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            # Make `latest-run` symlink to this run
            latest = self._base_runs_dir / "latest-run"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
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

    # Compatibility helpers used by tests
    def setup_run_directory(self, *, wandb_run) -> Path:
        """Initialize directories based on a W&B-like run object exposing .id.

        Creates the run directory, a checkpoints subdir, and updates latest-run symlink.
        """
        self._run_id = getattr(wandb_run, "id")
        self._run_dir = self._base_runs_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # Ensure checkpoints dir exists eagerly
        (self._run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        latest = self._base_runs_dir / "latest-run"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(Path(self._run_id))
        return self._run_dir

    def save_config(self, data: dict, filename: str = "config.json") -> Path:
        """Save a JSON config file under the run directory."""
        import json
        path = self.ensure_path(filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path