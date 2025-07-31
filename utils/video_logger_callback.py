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
    Watches a media directory for new files and logs them to Weights & Biases
    on epoch end. Fixed to compare absolute/real paths so under-root checks
    work reliably, and optionally handles writers that only emit 'modified'
    events.

    Parameters
    ----------
    media_root : str
        Directory that will contain media. If relative, it is resolved under
        the current W&B run directory (wandb.run.dir). If absolute, it is used
        as-is.
    exts : Iterable[str]
        File extensions to log (e.g., [".mp4", ".gif", ".png", ".jpg"]).
    namespace_depth : int
        Number of leading path parts (relative to media_root) to use as the W&B
        key (e.g., "train/<key>" or "eval/<key>").
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
        exts: Iterable[str] = (".mp4", ".gif", ".png", ".jpg"),
        namespace_depth: int = 2,
        include_modified: bool = True,
        min_age_sec: float = 0.25,
    ):
        # IMPORTANT: keep absolute paths absolute; don't strip leading slashes
        self.media_root = Path(media_root)

        self.exts = {e if e.startswith(".") else f".{e}" for e in exts}
        self.depth = max(1, int(namespace_depth))
        self.include_modified = bool(include_modified)
        self.min_age_sec = float(min_age_sec)

        self._seen: set[str] = set()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._observer = None
        self._handler = None
        self._root: Path | None = None

    # --------- path helpers --------------------------------------------------
    def _resolve_root(self, run_dir: str) -> Path:
        """Resolve the actual directory to watch/log under."""
        root = (
            self.media_root
            if self.media_root.is_absolute()
            else Path(run_dir) / self.media_root
        )
        return root.resolve()

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
    def _start_watch(self, root: Path):
        if self._observer:
            return

        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._root = root

        outer = self
        watched_root = root  # captured in handler closure

        class CreatedOrMovedInHandler(FileSystemEventHandler):
            def _maybe_add(self, p: Path):
                if p.suffix.lower() in outer.exts:
                    with outer._lock:
                        outer._pending.add(str(p.resolve()))

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

                dst_in = _is_under(dst, watched_root)
                src_in = _is_under(src, watched_root)
                if dst_in and not src_in:
                    self._maybe_add(dst)

            # Pick up writers that only modify an existing file.
            def on_modified(self, event):
                if not outer.include_modified or event.is_directory:
                    return
                self._maybe_add(Path(event.src_path))

        self._handler = CreatedOrMovedInHandler()
        self._observer = Observer()
        self._observer.schedule(self._handler, str(watched_root), recursive=True)
        self._observer.start()

    def _stop_watch(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = self._handler = self._root = None

    # --------- Lightning hooks ----------------------------------------------
    def on_fit_start(self, trainer, *_):
        # Start early so we don't miss files during the first epoch.
        run = getattr(wandb, "run", None)
        if run:
            self._start_watch(self._resolve_root(run.dir))

    def on_train_epoch_end(self, trainer, *_):
        self._process(trainer, "train")

    def on_validation_epoch_end(self, trainer, *_):
        self._process(trainer, "eval")

    def teardown(self, *a, **k):
        self._stop_watch()

    # --------- core ----------------------------------------------------------
    def _process(self, trainer, prefix: str):
        # Only proceed if a W&B run is active
        run = getattr(wandb, "run", None)
        if not run:
            return

        # Start watcher if it hasn't been started yet (e.g., validation-only run)
        if not self._observer:
            self._start_watch(self._resolve_root(run.dir))

        if self._root is None:
            return

        # Pull and clear any pending paths collected by the watcher
        with self._lock:
            paths = {Path(p) for p in self._pending}
            self._pending.clear()

        if not paths:
            return

        now = time.time()
        root = self._root.resolve()
        fresh: list[Path] = []

        for p in paths:
            try:
                rp = p.resolve()
                if str(rp) in self._seen:
                    continue
                if not rp.exists():
                    continue
                if not self._under_root(rp, root):
                    continue
                # age check to avoid reading partially-written files
                st = rp.stat()
                age = now - max(st.st_mtime, getattr(st, "st_ctime", st.st_mtime))
                if age < self.min_age_sec:
                    # Re-queue; next _process will pick it up
                    with self._lock:
                        self._pending.add(str(rp))
                    continue
                fresh.append(rp)
            except Exception:
                # Ignore transient races (e.g., file moved between stat and read)
                pass

        if not fresh:
            return

        # Group by key derived from relative path under root
        by_key: dict[str, list[Path]] = {}
        for p in sorted(fresh):
            rel = p.relative_to(root)
            key = self._key_for(rel)
            by_key.setdefault(key, []).append(p)

        # Log per key
        for key, files in by_key.items():
            media = [
                (wandb.Video if p.suffix.lower() in (".mp4", ".gif") else wandb.Image)(str(p))
                for p in files
            ]
            self._seen.update(map(str, files))
            trainer.logger.experiment.log({f"{prefix}/{key}": media})

    def _key_for(self, rel: Path) -> str:
        parts = rel.parts[: self.depth] or ("media",)
        return "/".join(parts)