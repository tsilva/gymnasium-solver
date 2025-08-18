from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, Iterable, Optional


class CsvMetricsLogger:
    """
    High-throughput CSV metrics logger.

    - Appends rows asynchronously using a background thread to avoid blocking training.
    - Uses a long-form schema to avoid costly header rewrites when metric keys vary.
      Columns: [time, total_timesteps, epoch, name, value]
    - Safe to call from the training hot path; enqueues work and returns immediately.
    """

    def __init__(self, path: str | Path, *, queue_size: int = 10000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Open file once; newline='' for correct CSV on all platforms
        self._fh = open(self.path, mode="a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)

        # Write header only if file is empty
        try:
            if self.path.stat().st_size == 0:
                self._writer.writerow(["time", "total_timesteps", "epoch", "name", "value"])
                self._fh.flush()
        except Exception:
            # If stat fails, proceed without header (degraded mode)
            pass

        # Async writer
        self._q: Queue[list] = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CsvMetricsLogger", daemon=True)
        self._thread.start()

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Enqueue metrics dict for writing.

        Metrics should contain numeric values. Non-numeric values are ignored.
        A single flush call may contain both train/* and eval/* namespaced keys.
        """
        if not metrics:
            return

        # Extract canonical step and epoch if available
        step = self._first_number(metrics.get("train/total_timesteps"), metrics.get("eval/total_timesteps"))
        epoch = self._first_number(metrics.get("train/epoch"), metrics.get("eval/epoch"))
        t = time.time()

        rows: list[list[Any]] = []
        for name, value in metrics.items():
            # Skip bookkeeping keys if desired, but keep them as metrics if numeric
            if not self._is_number(value):
                continue
            rows.append([t, step, epoch, name, float(value)])

        if not rows:
            return

        # Non-blocking put with drop-on-full to avoid stalling training
        try:
            self._q.put_nowait(rows)
        except Exception:
            # Queue full; drop this batch silently to preserve throughput
            pass

    def close(self, timeout: float = 2.0) -> None:
        """Signal writer to stop and close file."""
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass
        try:
            self._flush_remaining()
        finally:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass

    # -------- internals --------
    def _run(self) -> None:
        try:
            # Main loop; wake periodically to check stop flag
            while not self._stop.is_set():
                self._drain_once(timeout=0.1)
            # Final drain on stop
            self._drain_once(timeout=0.0)
        except Exception:
            # Never crash the process due to background writer errors
            pass

    def _drain_once(self, timeout: float) -> None:
        batched: list[list[Any]] = []
        try:
            rows = self._q.get(timeout=timeout)
            batched.extend(rows)
        except Empty:
            rows = None
        # Drain quickly any remaining items to write in a single syscall burst
        if rows is not None:
            while True:
                try:
                    more = self._q.get_nowait()
                    batched.extend(more)
                except Empty:
                    break
        if batched:
            self._writer.writerows(batched)
            try:
                self._fh.flush()
            except Exception:
                pass

    def _flush_remaining(self) -> None:
        # Write anything still in the queue
        remaining: list[list[Any]] = []
        while True:
            try:
                remaining.extend(self._q.get_nowait())
            except Empty:
                break
        if remaining:
            self._writer.writerows(remaining)
            try:
                self._fh.flush()
            except Exception:
                pass

    @staticmethod
    def _is_number(x: Any) -> bool:
        try:
            import numbers
            return isinstance(x, numbers.Number)
        except Exception:
            return False

    @staticmethod
    def _first_number(*values: Any) -> Optional[float]:
        for v in values:
            try:
                import numbers
                if isinstance(v, numbers.Number):
                    return float(v)
            except Exception:
                continue
        return None
