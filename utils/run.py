"""Run access helpers.

Encapsulates read-only access to a training run directory under `runs/`.

Usage:
    from utils.run import Run, list_run_ids
    run = Run.from_id("@latest-run")
    cfg = run.load_config()
    labels, mapping, default = run.checkpoint_choices()
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from utils.config import Config
from typing import Dict, List

from utils.io import read_json

LAST_RUN_DIR = Path("runs/@last") # TODO: use constant  
RUNS_DIR = Path("runs")

# TODO: move this to a more appropriate util location
def _symlink_to_dir(symlink_path: Path, target_dir: Path):
    if os.path.islink(symlink_path): os.unlink(symlink_path)
    symlink_path.symlink_to(target_dir.resolve(), target_is_directory=True)

def list_run_ids() -> List[str]:
    return [p.name for p in RUNS_DIR.iterdir() if p.is_dir()]

# TODO: rename to RunDir
@dataclass(frozen=True)
class Run:
    run_id: str
    run_dir: str
    read_only: bool

    @staticmethod
    def _resolve_run_dir(run_id: str) -> Path:
        return RUNS_DIR / run_id
    
    @classmethod
    def create(cls, run_id: str, config: Config) -> Run:
        # Ensure rundir exists
        run_dir = cls._resolve_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Symlink latest-run to this run dir (latest created)
        # TODO: extract symlink creation util
        if os.path.islink(LAST_RUN_DIR): os.unlink(LAST_RUN_DIR)
        LAST_RUN_DIR.symlink_to(run_dir.resolve(), target_is_directory=True)
        
        # Save config to run dir
        config_path = run_dir / "config.json"
        config.save_to_json(config_path)

        # Create and return run
        return cls(run_id, run_dir, read_only=False)

    @classmethod
    def load(cls, run_id: str) -> Run:
        run_dir = cls._resolve_run_dir(run_id)
        if not run_dir.exists(): raise FileNotFoundError(f"run directory not found: {run_dir}")
        return cls(run_id, run_dir, read_only=True)

    @property
    def id(self) -> str:
        assert self.run_id == self.run_dir.name, f"run_id {self.run_id} != run_dir.name {self.run_dir.name}"
        return self.run_id

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json" # TODO: use constant
        
    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv" # TODO: use constant

    def ensure_metrics_path(self) -> Path:
        return self._ensure_path(self.metrics_path)

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints" # TODO: use constant

    @property
    def video_dir(self) -> Path:
        return self.run_dir / "videos" # TODO: use constant

    def ensure_video_dir(self) -> Path:
        return self._ensure_path(self.video_dir)

    @property
    def last_checkpoint_dir(self) -> Path:
        return self.checkpoints_dir / "@last" # TODO: USE CONSTANT

    @property
    def best_checkpoint_dir(self) -> Path:
        return self.checkpoints_dir / "@best" # TODO: USE CONSTANT

    def load_config(self):
        data: Dict = read_json(self.config_path)
        return Config.build_from_dict(data)

    def _file_prefix_for_epoch(self, epoch: int) -> str:
        return f"epoch={epoch:02d}"

    def video_path_for_epoch(self, epoch: int) -> Path:
        file_prefix = self._file_prefix_for_epoch(epoch)
        return self.video_dir / f"{file_prefix}.mp4"

    def checkpoint_dir_for_epoch(self, epoch: int) -> Path:
        file_prefix = self._file_prefix_for_epoch(epoch)
        return self.checkpoints_dir / file_prefix

    def list_checkpoints(self) -> List[str]:
        return [p.name for p in self.checkpoints_dir.iterdir() if p.is_dir()]

    def save_checkpoint(self, epoch: int, source_dir: Path, is_best=False) -> None:
        # Move provided data to target checkpoint dir
        # (run manager doesn't need to know what this data is)
        checkpoint_dir = self.checkpoint_dir_for_epoch(epoch)
        shutil.move(source_dir, checkpoint_dir)

        # TODO: SOC violation; run needs to be aware that videos exist
        # TODO: it would be optimal if we only actually rendered videos from the best evals
        # Copy epoch files from video folder to checkpoint dir
        video_path = self.video_path_for_epoch(epoch)
        if os.path.exists(video_path): shutil.copy(video_path, checkpoint_dir)

        # Symlink as last checkpoint
        _symlink_to_dir(self.last_checkpoint_dir, checkpoint_dir)

        # If best checkpoint, symlink as best checkpoint
        if is_best: _symlink_to_dir(self.best_checkpoint_dir, checkpoint_dir)

    def _ensure_path(self, path: str | Path) -> Path:
        rel_path = Path(path)
        full_path = rel_path if self.run_dir in rel_path.parents else self.run_dir / rel_path
        target_dir = full_path if not full_path.suffix else full_path.parent
        if self.read_only and not target_dir.exists(): raise FileNotFoundError(f"Path not found: {target_dir}")
        else: target_dir.mkdir(parents=True, exist_ok=True)
        return full_path
