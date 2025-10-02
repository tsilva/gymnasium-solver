"""Run access helpers.

Encapsulates read-only access to a training run directory under `runs/`.

Usage:
    from utils.run import Run, list_run_ids
    run = Run.from_id("@last")
    cfg = run.load_config()
    labels, mapping, default = run.checkpoint_choices()
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.config import Config
from utils.io import read_json

RUNS_DIR = Path("runs")
LAST_RUN_DIR = RUNS_DIR / Path("@last")
CHECKPOINTS_DIR = Path("checkpoints")
LAST_CHECKPOINT_DIR = Path("@last")
BEST_CHECKPOINT_DIR = Path("@best")
CONFIG_FILENAME = Path("config.json")
RUN_LOG_FILENAME = Path("run.log")
METRICS_CSV_FILENAME = Path("metrics.csv")
POLICY_CHECKPOINT_FILENAME = Path("model.pt")
REGISTRY_FILENAME = RUNS_DIR / Path("runs.json")

# TODO: move this to a more appropriate util location
def _symlink_to_dir(symlink_path: Path, target_dir: Path):
    if os.path.islink(symlink_path): os.unlink(symlink_path)
    symlink_path.symlink_to(target_dir.resolve(), target_is_directory=True)

def list_run_ids() -> List[str]:
    return [p.name for p in RUNS_DIR.iterdir() if p.is_dir()]

def _read_registry() -> List[Dict[str, Any]]:
    """Read the runs registry from runs.json. Returns empty list if file doesn't exist."""
    if not REGISTRY_FILENAME.exists():
        return []
    with open(REGISTRY_FILENAME, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
            assert isinstance(data, list), f"Registry must be a list, got {type(data)}"
            return data
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def _write_registry(entries: List[Dict[str, Any]]) -> None:
    """Write the runs registry to runs.json with exclusive lock."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Sort by timestamp descending (newest first)
    sorted_entries = sorted(entries, key=lambda x: x["timestamp"], reverse=True)
    with open(REGISTRY_FILENAME, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(sorted_entries, f, indent=2)
            f.write("\n")
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def _add_run_to_registry(run_id: str, config: Config) -> None:
    """Add a run to the registry with metadata from config."""
    entries = _read_registry()

    # Remove existing entry if present (idempotent)
    entries = [e for e in entries if e["run_id"] != run_id]

    # Create new entry
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config_id": f"{config.env_id}:{config.algo_id}",
        "env_id": config.env_id,
        "algo": config.algo_id,
        "project_id": config.project_id,
    }
    entries.append(entry)

    _write_registry(entries)

def _remove_run_from_registry(run_id: str) -> None:
    """Remove a run from the registry."""
    entries = _read_registry()
    entries = [e for e in entries if e["run_id"] != run_id]
    _write_registry(entries)

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

        # Update registry
        _add_run_to_registry(run_id, config)

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
        return self.run_dir / CONFIG_FILENAME
        
    @property
    def metrics_path(self) -> Path:
        return self.run_dir / METRICS_CSV_FILENAME

    def ensure_metrics_path(self) -> Path:
        return self._ensure_path(self.metrics_path)

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / CHECKPOINTS_DIR

    @property
    def last_checkpoint_dir(self) -> Path:
        return self.checkpoints_dir / LAST_CHECKPOINT_DIR

    @property
    def best_checkpoint_dir(self) -> Path:
        return self.checkpoints_dir / BEST_CHECKPOINT_DIR

    @property
    def best_checkpoint_path(self) -> Path:
        return self._get_checkpoint_path(self.best_checkpoint_dir)

    @property
    def last_checkpoint_path(self) -> Path:
        return self._get_checkpoint_path(self.last_checkpoint_dir)

    def _get_checkpoint_path(self, checkpoint_dir: Path) -> Path:
        """Get the policy checkpoint path, trying both new and old formats."""
        # Try new format first (model.pt)
        new_path = checkpoint_dir / POLICY_CHECKPOINT_FILENAME
        if new_path.exists():
            return new_path

        # Fall back to old format (policy.ckpt) for backward compatibility
        old_path = checkpoint_dir / "policy.ckpt"
        if old_path.exists():
            return old_path

        # If neither exists, return the new format path (will fail later with clear error)
        return new_path

    def load_config(self):
        data: Dict = read_json(self.config_path)
        return Config.build_from_dict(data)

    def _file_prefix_for_epoch(self, epoch: int) -> str:
        return f"epoch={epoch:02d}"

    def checkpoint_dir_for_epoch(self, epoch: int) -> Path:
        file_prefix = self._file_prefix_for_epoch(epoch)
        return self.checkpoints_dir / file_prefix

    def video_path_for_epoch(self, epoch: int) -> Path:
        file_prefix = self._file_prefix_for_epoch(epoch)
        return self.checkpoint_dir_for_epoch(epoch) / f"{file_prefix}.mp4"

    def list_checkpoints(self) -> List[str]:
        return [p.name for p in self.checkpoints_dir.iterdir() if p.is_dir()]

    def save_checkpoint(self, epoch: int, source_dir: Path, is_best=False) -> None:
        # Move provided data to target checkpoint dir
        checkpoint_dir = self.checkpoint_dir_for_epoch(epoch)
        shutil.move(source_dir, checkpoint_dir)

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

    def delete(self) -> None:
        """Delete the run directory and remove from registry."""
        assert not self.read_only, "Cannot delete a read-only run"

        # Update @last symlink if it points to this run
        if LAST_RUN_DIR.exists() and LAST_RUN_DIR.resolve() == self.run_dir.resolve():
            os.unlink(LAST_RUN_DIR)

        # Remove from registry
        _remove_run_from_registry(self.run_id)

        # Delete run directory
        shutil.rmtree(self.run_dir)
