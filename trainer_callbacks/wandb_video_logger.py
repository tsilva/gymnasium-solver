"""Video logger callback for watching and logging media files to Weights & Biases."""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pytorch_lightning as pl
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import wandb


@dataclass
class _PhaseState:
    pending: set[str] = field(default_factory=set)
    root: Path | None = None


# TODO: REFACTOR this file
class WandbVideoLoggerCallback(pl.Callback):
    """Watch media dirs for new videos/images and log them to Weights & Biases on epoch end."""

    PHASE_DIRS: dict[str, str] = {"train": "train", "val": "val"}

    def __init__(
        self,
        *,
        media_root: str = "videos",
        exts: Iterable[str] = [".mp4"],
        namespace_depth: int = 1,
        include_modified: bool = True,
        min_age_sec: float = 0.0 # TODO: was previously 0.25, but last video is not recorded with that value
    ):
        # IMPORTANT: keep absolute paths absolute; don't strip leading slashes
        self.media_root = Path(media_root)

        self.exts = {e if e.startswith(".") else f".{e}" for e in exts}
        self.depth = max(1, int(namespace_depth))
        self.include_modified = bool(include_modified)
        self.min_age_sec = float(min_age_sec)

        self._seen: set[str] = set()
        self._phases: dict[str, _PhaseState] = {
            name: _PhaseState() for name in self.PHASE_DIRS
        }
        self._lock = threading.Lock()
        self._observer = None
        self._handler = None
        self._trainer = None

    # --------- path helpers --------------------------------------------------
    def _resolve_roots(self, run_dir: str) -> dict[str, Path]:
        """Resolve the actual directories to watch/log under for each phase."""
        base_root = (
            self.media_root
            if self.media_root.is_absolute()
            else Path(run_dir) / self.media_root
        )

        return {
            name: (base_root / dir_name).resolve()
            for name, dir_name in self.PHASE_DIRS.items()
        }

    def _under_root(self, p: Path, root: Path) -> bool:
        """Return True if path p is under root (both resolved/absolute)."""
        p = Path(p).resolve()
        root = Path(root).resolve()
        try:
            p.relative_to(root)
            return True
        except ValueError:
            return False

    def _set_roots(self, roots: dict[str, Path]):
        for name, root in roots.items():
            state = self._phases.get(name)
            if state:
                state.root = Path(root).resolve()

    def _phase_for_path(self, path: Path) -> str | None:
        resolved = Path(path).resolve()
        for name, state in self._phases.items():
            root = state.root
            if root and self._under_root(resolved, root):
                return name
        return None

    def _add_pending(self, prefix: str, path: Path):
        state = self._phases.get(prefix)
        if not state:
            return
        with self._lock:
            state.pending.add(str(Path(path).resolve()))

    def _queue_pending(self, path: Path):
        prefix = self._phase_for_path(path)
        if prefix:
            self._add_pending(prefix, path)

    def _enqueue_existing_media(self):
        with self._lock:
            for name, state in self._phases.items():
                root = state.root
                if not root or not root.exists():
                    continue
                for p in root.rglob("*"):
                    if p.is_file() and p.suffix.lower() in self.exts:
                        state.pending.add(str(p.resolve()))

    # --------- watchdog wiring ----------------------------------------------
    def _start_watch(self, roots: dict[str, Path]):
        if self._observer:
            return

        # If watchdog isn't available, skip live watching; we'll scan on epoch end
        if Observer is None or FileSystemEventHandler is None:
            self._set_roots({name: Path(root).resolve() for name, root in roots.items()})
            return

        resolved_roots = {name: Path(root).resolve() for name, root in roots.items()}
        self._set_roots(resolved_roots)

        # Do NOT create directories here; only watch ones that already exist
        exists_map = {name: root.exists() for name, root in resolved_roots.items()}
        if not any(exists_map.values()):
            return

        outer = self

        class CreatedOrMovedInHandler(FileSystemEventHandler):
            def _maybe_add(self, p: Path):
                if p.suffix.lower() in outer.exts:
                    outer._queue_pending(p.resolve())

            def on_created(self, event):
                if event.is_directory:
                    return
                self._maybe_add(Path(event.src_path))

            # Treat files moved into the watched root as "new" (atomic writes).
            def on_moved(self, event):
                if event.is_directory:
                    return
                src = Path(event.src_path).resolve()
                dst = Path(event.dest_path).resolve()
                dst_phase = outer._phase_for_path(dst)
                src_phase = outer._phase_for_path(src)

                if dst_phase and dst_phase != src_phase:
                    self._maybe_add(dst)

            # Pick up writers that only modify an existing file.
            def on_modified(self, event):
                if not outer.include_modified or event.is_directory:
                    return
                self._maybe_add(Path(event.src_path))

        self._handler = CreatedOrMovedInHandler()
        self._observer = Observer()

        # Watch existing directories only
        scheduled = False
        for name, root in resolved_roots.items():
            if exists_map.get(name):
                self._observer.schedule(self._handler, str(root), recursive=True)
                scheduled = True

        # Start observer only if at least one directory is scheduled
        if scheduled:
            self._observer.start()
        else:
            # Clean up if nothing to watch
            self._observer = None
            self._handler = None

    def _stop_watch(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._handler = None
        for state in self._phases.values():
            state.root = None

    # --------- Lightning hooks ----------------------------------------------
    def on_fit_start(self, trainer, *_):
        # Store trainer reference for later use
        self._trainer = trainer
        # Start early so we don't miss files during the first epoch.
        run = getattr(wandb, "run", None)
        if run:
            roots = self._resolve_roots(run.dir)
            self._start_watch(roots)

    def on_train_epoch_end(self, trainer, *_):
        self._process(trainer, "train")

    def on_validation_epoch_end(self, trainer, *_):
        self._process(self._trainer, "val")

    def on_fit_end(self, trainer, *_):
        # Process any remaining videos before training ends
        self._process(trainer, "train")
        self._process(trainer, "val")

    def teardown(self, *a, **k):
        # Process any remaining videos one last time before shutdown
        if self._trainer:
            self._process(self._trainer, "train")
            self._process(self._trainer, "val")
        self._stop_watch()

    # --------- core ----------------------------------------------------------
    def _process(self, trainer, prefix: str):
        if trainer is None:
            return
        # Only proceed if a W&B run is active
        run = getattr(wandb, "run", None)
        if not run:
            return

        # Start watcher if it hasn't been started yet (e.g., validation-only run)
        if not self._observer:
            roots = self._resolve_roots(run.dir)
            self._start_watch(roots)
            self._enqueue_existing_media()

        state = self._phases.get(prefix)
        if not state or state.root is None:
            return

        # Pull and clear pending paths for the appropriate phase
        with self._lock:
            paths = {Path(p) for p in state.pending}
            state.pending.clear()
            current_root = state.root

        if not paths:
            return

        now = time.time()
        current_root = current_root.resolve()
        fresh: list[Path] = []

        for p in paths:
            rp = p.resolve()
            if str(rp) in self._seen:
                continue
            if not rp.exists():
                continue
            if not self._under_root(rp, current_root):
                continue
            # age check to avoid reading partially-written files
            st = rp.stat()
            age = now - max(st.st_mtime, getattr(st, "st_ctime", st.st_mtime))
            if age < self.min_age_sec:
                # Re-queue; next _process will pick it up
                self._add_pending(prefix, rp)
                continue
            fresh.append(rp)

        if not fresh:
            return

        # Group by key derived from relative path under current root
        by_key: dict[str, list[Path]] = {}
        for p in sorted(fresh):
            rel = p.relative_to(current_root)
            key = self._key_for(rel)
            by_key.setdefault(key, []).append(p)

        # Log per key using Lightning's proper logging mechanism
        for key, files in by_key.items():
            media = []
            for p in files:
                # Determine format from file extension
                ext = p.suffix.lower()
                if ext == '.mp4':
                    format_type = "mp4"
                elif ext == '.gif':
                    format_type = "gif"
                elif ext in ['.png', '.jpg', '.jpeg']:
                    # For images, use wandb.Image instead
                    img_format = "png" if ext == '.png' else "jpg"
                    media.append(wandb.Image(str(p), format=img_format))
                    continue
                else:
                    # Default to mp4 for unknown video extensions
                    format_type = "mp4"
                
                media.append(wandb.Video(str(p), format=format_type))
            
            self._seen.update(map(str, files))
            # Try to get total_timesteps from the trainer's pl_module for better step alignment
            step_value = trainer.global_step
            if hasattr(trainer, 'lightning_module') and hasattr(trainer.lightning_module, 'train_collector'):
                metrics = trainer.lightning_module.train_collector.get_metrics()
                from utils.metrics_config import metrics_config
                step_key = metrics_config.total_timesteps_key()
                # Resolve unnamespaced variant of the step key
                # e.g., 'train/cnt/total_timesteps' -> 'cnt/total_timesteps'
                try:
                    _, bare = step_key.split("/", 1)
                except Exception:
                    bare = "cnt/total_timesteps"
                step_value = metrics.get(bare, trainer.global_step)
            
            # Use Lightning's log_metrics to ensure proper step handling
            trainer.logger.log_metrics({f"{prefix}/{key}": media}, step=step_value)

    def _key_for(self, rel: Path) -> str:
        parts = rel.parts[: self.depth] or ("media",)
        return "/".join(parts)
