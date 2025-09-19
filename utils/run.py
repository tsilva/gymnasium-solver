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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.config import Config
from utils.io import read_json

RUNS_DIR = Path("runs")
LATEST_SYMLINK_NAMES = ("@latest-run")
BEST_CKPT_NAMES = {"best.ckpt"}
LAST_CKPT_NAMES = {"last.ckpt"}


def list_run_ids(runs_dir: Path = RUNS_DIR) -> List[str]:
    """Return run directory names sorted by modified time (newest first)."""

    if not runs_dir.exists():
        return []

    run_dirs = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name not in {"@latest-run", "latest-run"}
    ]

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in run_dirs]

@dataclass(frozen=True)
class Run:
    run_dir: Path = RUNS_DIR
    checkpoints_dir: Path

    def __init__(self, 
        run_id: str,
        config: Config,
        *,
        ensure_checkpoints: bool = True,
        update_latest_symlink: bool = True
    ) -> None:
        # Save configuration to run directory
        config_path = self._ensure_path("config.json")
        config.save_to_json(config_path)

    @staticmethod
    def _set_latest_symlink(runs_root: Path, run_id: str) -> None:
        runs_root.mkdir(parents=True, exist_ok=True)
        for link_name in LATEST_SYMLINK_NAMES:
            link_path = runs_root / link_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(Path(run_id))

    @property
    def id(self) -> str:
        return self.run_dir.name

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"
        
    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    # Compatibility helpers for call-sites still using RunManager-style APIs
    def get_run_dir(self) -> Path:
        return self.run_dir

    def get_run_id(self) -> str:
        return self.id

    def _ensure_path(self, path: str | Path) -> Path:
        """Ensure that the relative path (or directory) exists under the run."""

        rel_path = Path(path)
        full_path = self.run_dir / rel_path
        target_dir = full_path if not full_path.suffix else full_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return full_path

    def ensure_latest_symlink(self) -> None:
        """Force-update the latest-run symlink to point at this run."""

        self._set_latest_symlink(self.runs_root, self.id)

    def load_config(self):
        """Load the run's config into a utils.config.Config instance."""
        from utils.config import Config
        data: Dict = read_json(self.config_path)
        return Config.build_from_dict(data)

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Return the best checkpoint path when present."""
        for name in BEST_CKPT_NAMES:
            candidate = self.checkpoints_dir / name
            if candidate.exists():
                return candidate
        return None

    def checkpoint_choices(self) -> Tuple[List[str], Dict[str, Path], Optional[str]]:
        """Return (labels, mapping, default_label) for available checkpoints."""

        ckpt_dir = self.checkpoints_dir
        if not ckpt_dir.exists():
            return [], {}, None

        files = [p for p in ckpt_dir.glob("*.ckpt") if p.is_file()]
        if not files:
            return [], {}, None

        def _score(path: Path) -> Tuple[int, float]:
            name = path.name
            if name in BEST_CKPT_NAMES:
                return (0, -path.stat().st_mtime)
            if name in LAST_CKPT_NAMES:
                return (1, -path.stat().st_mtime)
            return (2, -path.stat().st_mtime)

        files.sort(key=_score)

        labels: List[str] = []
        mapping: Dict[str, Path] = {}
        for path in files:
            label = path.name
            if label in BEST_CKPT_NAMES:
                label = f"{label} (best)"
            elif label in LAST_CKPT_NAMES:
                label = f"{label} (last)"
            labels.append(label)
            mapping[label] = path

        default_label = None
        for label in labels:
            if label.startswith("best"):
                default_label = label
                break
        if default_label is None:
            default_label = labels[0]

        return labels, mapping, default_label

    def resolve_checkpoint(self, label: Optional[str]) -> Path:
        """Resolve a checkpoint label or filename to a concrete checkpoint path."""

        labels, mapping, default_label = self.checkpoint_choices()
        if not mapping:
            raise FileNotFoundError(f"No checkpoints found under: {self.checkpoints_dir}")

        if label is None:
            assert default_label is not None
            return mapping[default_label]

        if label in mapping:
            return mapping[label]

        # Accept raw filenames (without the "(best)/(last)" suffix)
        for display_label, path in mapping.items():
            if Path(display_label).name == label:
                return path

        candidate = self.checkpoints_dir / label
        if candidate.exists():
            return candidate

        raise ValueError(f"Unknown checkpoint label '{label}'. Available: {list(mapping.keys())}")

