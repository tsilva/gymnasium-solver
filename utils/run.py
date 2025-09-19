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

from utils.io import read_json, write_json

RUNS_ROOT = Path("runs")
LATEST_SYMLINK_NAMES = ("@latest-run", "latest-run")
BEST_CKPT_NAMES = {"best.ckpt", "best_checkpoint.ckpt"}
LAST_CKPT_NAMES = {"last.ckpt", "last_checkpoint.ckpt"}


def _resolve_run_dir(run_id: str, runs_dir: Path = RUNS_ROOT) -> Path:
    """Resolve a run identifier (including @latest-run) to a concrete directory."""

    if run_id in {"@latest-run", "latest-run"}:
        for link_name in LATEST_SYMLINK_NAMES:
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
    runs_root: Path = RUNS_ROOT

    @staticmethod
    def _normalise_root(runs_root: Path | str) -> Path:
        root = Path(runs_root)
        return root if root.is_absolute() else Path(root)

    @classmethod
    def create(
        cls,
        run_id: str,
        runs_root: Path | str = RUNS_ROOT,
        *,
        ensure_checkpoints: bool = True,
        update_latest_symlink: bool = True,
    ) -> "Run":
        """Instantiate a Run for `run_id`, ensuring required directories exist.

        The returned Run points at ``runs_root/run_id`` and optionally updates the
        ``@latest-run`` symlink to reference the new run.
        """

        root = cls._normalise_root(runs_root)
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if ensure_checkpoints:
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        if update_latest_symlink:
            cls._set_latest_symlink(root, run_id)
        return cls(run_dir=run_dir, runs_root=root)

    @classmethod
    def from_id(cls, run_id: str, runs_root: Path | str = RUNS_ROOT) -> "Run":
        run_path = _resolve_run_dir(run_id, Path(runs_root))
        return cls(run_dir=run_path, runs_root=Path(runs_root))

    @classmethod
    def from_wandb_run(cls, *, wandb_run, runs_root: Path | str = RUNS_ROOT) -> "Run":
        run_id = getattr(wandb_run, "id")
        if not run_id:
            raise ValueError("wandb_run.id must be set to create a Run")
        return cls.create(run_id, runs_root=runs_root)

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
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    # Compatibility helpers for call-sites still using RunManager-style APIs
    def get_run_dir(self) -> Path:
        return self.run_dir

    def get_run_id(self) -> str:
        return self.id

    def ensure_path(self, path: str | Path) -> Path:
        """Ensure that the relative path (or directory) exists under the run."""

        rel_path = Path(path)
        full_path = self.run_dir / rel_path
        target_dir = full_path if not full_path.suffix else full_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return full_path

    def ensure_latest_symlink(self) -> None:
        """Force-update the latest-run symlink to point at this run."""

        self._set_latest_symlink(self.runs_root, self.id)

    def save_json(self, data: Dict, filename: str = "config.json") -> Path:
        path = self.ensure_path(filename)
        write_json(path, data)
        return path

    def save_config(self, data: Dict, filename: str = "config.json") -> Path:
        """Backward-compatible wrapper around :meth:`save_json`."""

        return self.save_json(data, filename)

    @property
    def config_path(self) -> Path:
        """Return config.json path under the run directory."""
        p = self.run_dir / "config.json"
        if p.exists():
            return p
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

    
