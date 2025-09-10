"""Run access helpers.

Encapsulates read-only access to a training run directory under `runs/`.

Usage:
    from utils.run import Run
    run = Run.from_id("@latest-run")
    cfg = run.load_config()
    ckpt = run.default_checkpoint
    path = run.dir
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

@dataclass(frozen=True)
class Run:
    run_dir: Path

    @staticmethod
    def from_id(run_id: str) -> "Run":
        run_path = Path("runs") / run_id
        if not run_path.exists(): raise FileNotFoundError(f"Run directory not found: {run_path}")
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
        import json
        from utils.config import Config
        with open(self.config_path, "r", encoding="utf-8") as f: data: Dict = json.load(f)
        return Config.build_from_dict(data)

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Return the best checkpoint path when present."""
        p = self.checkpoints_dir / "best.ckpt"
        if p.exists(): return p
        return None

    @property
    def last_checkpoint_path(self) -> Optional[Path]:
        """Return the last checkpoint path when present."""
        p = self.checkpoints_dir / "last.ckpt"
        if p.exists(): return p
        return None
