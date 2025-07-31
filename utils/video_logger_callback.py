import time
from pathlib import Path
from typing import Iterable, Dict, List, Optional

import wandb
import pytorch_lightning as pl


# TODO: use watchdog
class VideoLoggerCallback(pl.Callback):
    """
    PyTorch Lightning callback that automatically logs video files found under
    <wandb.run.dir>/<media_root>/<namespace>/<name>/*.{mp4,gif,png,jpg}
    as W&B Media with dashboard keys derived from the directory path.
    
    Logs videos as:
    - "train/{key}" during on_train_epoch_end
    - "eval/{key}" during on_validation_epoch_end
    """
    
    def __init__(
        self,
        *,
        media_root: str = "videos",
        exts: Iterable[str] = (".mp4", ".gif", ".png", ".jpg"),
        namespace_depth: int = 2,
        log_interval_s: float = 10.0,
        max_per_key: int = 16,
    ):
        """
        Args:
            media_root: top-level folder under wandb.run.dir to scan.
            exts: file extensions to log (lowercased).
            namespace_depth: how many path components under media_root build the metric key.
                             2 => logs "namespace/name".
            log_interval_s: throttle for directory scanning.
            max_per_key: cap how many new media items to log per key per scan.
        """
        super().__init__()
        self.media_root = media_root.strip("/")
        self.exts = tuple(e if e.startswith(".") else f".{e}" for e in exts)
        self.namespace_depth = max(1, int(namespace_depth))
        self.log_interval_s = float(log_interval_s)
        self.max_per_key = int(max_per_key)

        self._last_sync = 0.0
        self._seen: set[str] = set()
        self._size_cache: Dict[str, int] = {}

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._scan_and_log(trainer, prefix="train")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._scan_and_log(trainer, prefix="eval")

    def _scan_and_log(self, trainer,  prefix: str) -> None:
        """Scan for new videos and log them with the given prefix."""
        if not hasattr(wandb, 'run') or wandb.run is None:
            return

        root = Path(wandb.run.dir) / self.media_root
        if not root.exists():
            return

        files_by_key = self._gather_new_files(root)
        
        for key, files in files_by_key.items():
            if not files:
                continue
            
            # Limit spam
            files = files[-self.max_per_key:]
            media = []
            
            for f in files:
                if f.suffix.lower() in (".mp4", ".gif"):
                    media.append(wandb.Video(str(f), format=f.suffix.lstrip(".")))
                else:
                    media.append(wandb.Image(str(f)))
                self._seen.add(str(f))
            
            # Log directly to wandb with prefix since pl_module.log_dict can't handle wandb objects
            full_key = f"{prefix}/{key}"
            trainer.logger.experiment.log({full_key: media})
            
    def _key_for(self, rel_path: Path) -> str:
        """
        Map a file path relative to <run.dir>/<media_root> to a W&B key.
        E.g., videos/train/rollouts/clip.mp4 -> "train/rollouts" when namespace_depth=2.
        """
        parts = rel_path.parts[:self.namespace_depth]
        return "/".join(parts) if parts else "media"

    def _gather_new_files(self, root: Path) -> Dict[str, List[Path]]:
        """
        Find not-yet-logged, size-stable files and bucket them by derived key.
        """
        by_key: Dict[str, List[Path]] = {}
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in self.exts:
                continue
            sp = str(p)

            # Size-stability check to avoid logging files that are still being written.
            size = p.stat().st_size
            last = self._size_cache.get(sp)
            self._size_cache[sp] = size
            if last is not None and last != size:
                # skip this pass; will pick it up when size stabilizes
                continue

            if sp in self._seen:
                continue

            rel = p.relative_to(root)
            key = self._key_for(rel)
            by_key.setdefault(key, []).append(p)

        return by_key
