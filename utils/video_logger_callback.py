import threading
import time
from pathlib import Path
from typing import Iterable

import wandb
import pytorch_lightning as pl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class VideoLoggerCallback(pl.Callback):
    """
    Watches multiple media directories for new files and logs them to Weights & Biases
    on epoch end. Supports separate directories for train and eval videos.

    Parameters
    ----------
    media_root : str
        Base directory that will contain media subdirectories. If relative, it is 
        resolved under the current W&B run directory (wandb.run.dir). If absolute, 
        it is used as-is.
    exts : Iterable[str]
        File extensions to log (e.g., [".mp4", ".gif", ".png", ".jpg"]).
    namespace_depth : int
        Number of leading path parts (relative to media_root/train or media_root/eval) 
        to use as the W&B key (e.g., "episodes" for videos in train/episodes/ or eval/episodes/).
    include_modified : bool
        If True, also treat file modifications as "new" (useful for writers
        that stream bytes to an already-created file).
    min_age_sec : float
        Minimum file age before logging (helps avoid reading partially-written
        files).
    """

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
        self._pending_train: set[str] = set()
        self._pending_eval: set[str] = set()
        self._lock = threading.Lock()
        self._observer = None
        self._handler = None
        self._train_root: Path | None = None
        self._eval_root: Path | None = None
        self._trainer = None

    # --------- path helpers --------------------------------------------------
    def _resolve_roots(self, run_dir: str) -> tuple[Path, Path]:
        """Resolve the actual directories to watch/log under for train and eval."""
        base_root = (
            self.media_root
            if self.media_root.is_absolute()
            else Path(run_dir) / self.media_root
        )
        
        train_root = (base_root / "train").resolve()
        eval_root = (base_root / "eval").resolve()
        
        return train_root, eval_root

    def _under_root(self, p: Path, root: Path) -> bool:
        """Return True if path p is under root (both resolved/absolute)."""
        p = Path(p).resolve()
        root = Path(root).resolve()
        try:
            p.relative_to(root)
            return True
        except ValueError:
            return False

    # --------- watchdog wiring ----------------------------------------------
    def _start_watch(self, train_root: Path, eval_root: Path):
        if self._observer:
            return

        train_root = train_root.resolve()
        eval_root = eval_root.resolve()
        
        # Create directories if they don't exist
        train_root.mkdir(parents=True, exist_ok=True)
        eval_root.mkdir(parents=True, exist_ok=True)
        
        self._train_root = train_root
        self._eval_root = eval_root

        outer = self
        
        class CreatedOrMovedInHandler(FileSystemEventHandler):
            def _maybe_add(self, p: Path):
                if p.suffix.lower() in outer.exts:
                    resolved_p = p.resolve()
                    with outer._lock:
                        # Determine which pending set to add to based on path
                        if outer._under_root(resolved_p, train_root):
                            outer._pending_train.add(str(resolved_p))
                        elif outer._under_root(resolved_p, eval_root):
                            outer._pending_eval.add(str(resolved_p))

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

                def _is_under(path: Path, base: Path) -> bool:
                    try:
                        path.relative_to(base)
                        return True
                    except ValueError:
                        return False

                # Check if moved into either train or eval roots
                dst_in_train = _is_under(dst, train_root)
                src_in_train = _is_under(src, train_root)
                dst_in_eval = _is_under(dst, eval_root)
                src_in_eval = _is_under(src, eval_root)
                
                if (dst_in_train and not src_in_train) or (dst_in_eval and not src_in_eval):
                    self._maybe_add(dst)

            # Pick up writers that only modify an existing file.
            def on_modified(self, event):
                if not outer.include_modified or event.is_directory:
                    return
                self._maybe_add(Path(event.src_path))

        self._handler = CreatedOrMovedInHandler()
        self._observer = Observer()
        
        # Watch both train and eval directories
        self._observer.schedule(self._handler, str(train_root), recursive=True)
        self._observer.schedule(self._handler, str(eval_root), recursive=True)
        self._observer.start()

    def _stop_watch(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = self._handler = self._train_root = self._eval_root = None

    # --------- Lightning hooks ----------------------------------------------
    def on_fit_start(self, trainer, *_):
        # Store trainer reference for later use
        self._trainer = trainer
        # Start early so we don't miss files during the first epoch.
        run = getattr(wandb, "run", None)
        if run:
            train_root, eval_root = self._resolve_roots(run.dir)
            self._start_watch(train_root, eval_root)

    def on_train_epoch_end(self, trainer, *_):
        self._process(trainer, "train")

    def on_validation_epoch_end(self, trainer, *_):
        # Videos are now processed immediately during validation_step
        # to ensure correct timestep alignment with eval metrics
        pass

    def on_fit_end(self, trainer, *_):
        # Process any remaining videos before training ends
        self._process(trainer, "train")
        self._process(trainer, "eval")

    def teardown(self, *a, **k):
        # Process any remaining videos one last time before shutdown
        if self._trainer:
            self._process(self._trainer, "train")
            self._process(self._trainer, "eval")
        self._stop_watch()

    # --------- core ----------------------------------------------------------
    def _process(self, trainer, prefix: str):
        # Only proceed if a W&B run is active
        run = getattr(wandb, "run", None)
        if not run:
            return

        # Start watcher if it hasn't been started yet (e.g., validation-only run)
        if not self._observer:
            train_root, eval_root = self._resolve_roots(run.dir)
            self._start_watch(train_root, eval_root)

        if self._train_root is None or self._eval_root is None:
            return

        # Pull and clear pending paths for the appropriate phase
        with self._lock:
            if prefix == "train":
                paths = {Path(p) for p in self._pending_train}
                self._pending_train.clear()
                current_root = self._train_root
            else:  # eval
                paths = {Path(p) for p in self._pending_eval}
                self._pending_eval.clear()
                current_root = self._eval_root

        if not paths:
            return

        now = time.time()
        current_root = current_root.resolve()
        fresh: list[Path] = []

        for p in paths:
            try:
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
                    with self._lock:
                        if prefix == "train":
                            self._pending_train.add(str(rp))
                        else:
                            self._pending_eval.add(str(rp))
                    continue
                fresh.append(rp)
            except Exception:
                # Ignore transient races (e.g., file moved between stat and read)
                pass

        if not fresh:
            return

        # Group by key derived from relative path under current root
        by_key: dict[str, list[Path]] = {}
        for p in sorted(fresh):
            rel = p.relative_to(current_root)
            key = self._key_for(rel)
            by_key.setdefault(key, []).append(p)

        # Log per key
        for key, files in by_key.items():
            media = [wandb.Video(str(p)) for p in files]
            self._seen.update(map(str, files))
            trainer.logger.experiment.log({f"{prefix}/{key}": media})

    def _key_for(self, rel: Path) -> str:
        parts = rel.parts[: self.depth] or ("media",)
        return "/".join(parts)