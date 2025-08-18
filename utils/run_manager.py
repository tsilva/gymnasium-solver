"""Run directory management utilities for organizing all run assets."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import wandb


class RunManager:
    """Manages run-specific directories and assets organization."""

    def __init__(self, base_runs_dir: str = "runs"):
        self.base_runs_dir = Path(base_runs_dir)
        self.run_dir: Optional[Path] = None
        self.run_id: Optional[str] = None

    def setup_run_directory(self, wandb_run: Optional[object] = None) -> Path:
        """Create the run directory structure using the W&B run id."""
        if wandb_run is None:
            wandb_run = wandb.run
        if wandb_run is None:
            raise ValueError("No wandb run available. Make sure wandb.init() has been called.")

        self.run_id = wandb_run.id
        self.run_dir = self.base_runs_dir / self.run_id

        # Create dirs
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "videos").mkdir(exist_ok=True)

        self._update_latest_run_symlink()
        return self.run_dir

    def _update_latest_run_symlink(self) -> None:
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        latest = self.base_runs_dir / "latest-run"
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                pass
        latest.symlink_to(Path(self.run_id))

    def get_checkpoint_dir(self) -> Path:
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir / "checkpoints"

    def get_video_dir(self) -> Path:
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir / "videos"

    def get_logs_dir(self) -> Path:
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        # Store logs directly at the run root (run.log), no logs subfolder
        return self.run_dir

    def get_configs_dir(self) -> Path:
        """Compatibility: place configs at run root."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir

    def save_config(self, config, filename: str = "config.json") -> Path:
        """Write config.json at the run root."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        path = self.run_dir / filename
        if hasattr(config, "__dataclass_fields__"):
            data = asdict(config)
        else:
            data = config
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def get_run_info(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "base_runs_dir": str(self.base_runs_dir),
        }
