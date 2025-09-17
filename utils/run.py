"""Run access helpers.

Encapsulates read-only access to a training run directory under `runs/`.

Usage:
    from utils.run import Run, list_run_ids
    run = Run.from_id("@latest-run")
    cfg = run.load_config()
    labels, mapping, default = run.checkpoint_choices()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.io import read_json


RUNS_ROOT = Path("runs")
BEST_CKPT_NAMES = {"best.ckpt", "best_checkpoint.ckpt"}
LAST_CKPT_NAMES = {"last.ckpt", "last_checkpoint.ckpt"}


def _resolve_run_dir(run_id: str, runs_dir: Path = RUNS_ROOT) -> Path:
    """Resolve a run identifier (including @latest-run) to a concrete directory."""

    if run_id in {"@latest-run", "latest-run"}:
        for link_name in ("@latest-run", "latest-run"):
            link_path = runs_dir / link_name
            if link_path.is_symlink():
                target = link_path.resolve()
                if target.exists():
                    return target
            elif link_path.exists():
                return link_path
        raise FileNotFoundError("@latest-run symlink not found")

    run_path = runs_dir / run_id
    if run_path.is_symlink():
        resolved = run_path.resolve()
        if resolved.exists():
            return resolved
    if run_path.exists():
        return run_path
    raise FileNotFoundError(f"Run directory not found: {run_path}")


def list_run_ids(runs_dir: Path = RUNS_ROOT) -> List[str]:
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
    run_dir: Path

    @staticmethod
    def from_id(run_id: str) -> "Run":
        run_path = _resolve_run_dir(run_id)
        return Run(run_dir=run_path)

    @property
    def id(self) -> str:
        return self.run_dir.name

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def config_path(self) -> Path:
        """Return config.json path under the run directory."""
        p = self.run_dir / "config.json"
        if p.exists(): return p
        raise FileNotFoundError(f"config.json not found under run: {self.run_dir}")

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

    
