from __future__ import annotations

import csv
import numbers
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional

from utils.metrics_config import metrics_config


# TODO: refactor this file
class MetricsCSVLogger:
    """
    High-throughput CSV metrics logger (wide format).

    - Appends rows asynchronously using a background thread to avoid blocking training.
    - Writes wide-form CSV with metric names as individual columns.
      Columns: [time, total_timesteps, epoch, <metric1>, <metric2>, ...]
    - Dynamically upgrades the header when new metric keys appear, using an atomic
      file rewrite to preserve previous rows.
    - Detects legacy long-form files (columns: name, value) and rotates them to
      a .legacy.csv file, starting fresh in wide format.
    - Safe to call from the training hot path; enqueues work and returns immediately.
    """

    def __init__(self, path: str | Path, *, queue_size: int = 10000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Base fields always first
        self._base_fields = ["time", "total_timesteps", "epoch"]
        self._ignore_legacy_fields = {"name", "value"}
        self._fieldnames = list(self._base_fields)

        # If file exists with a header, adopt it unless it's legacy long-form
        header = self._read_existing_header(self.path)
        if header is not None:
            header_set = set(x.strip() for x in header)
            if {"name", "value"}.issubset(header_set):
                # Rotate legacy file and start fresh in wide format
                self._rotate_legacy_file()
                self._init_new_file_with_header(self._fieldnames)
            else:
                # Adopt existing header
                self._fieldnames = list(header)
                self._fh = open(self.path, mode="a", encoding="utf-8", newline="")
                self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        else:
            # Create new file with base header only
            self._init_new_file_with_header(self._fieldnames)

        # Async writer
        self._q = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="CsvMetricsLogger", daemon=True)
        self._thread.start()

    def buffer_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Enqueue metrics dict for writing.

        Metrics should contain numeric values. Non-numeric values are ignored.
        A single flush call may contain both train/* and eval/* namespaced keys.
        """
        if not metrics:
            return

        # Extract canonical step and epoch if available
        step = self._first_number(metrics.get("train/total_timesteps"), metrics.get("val/total_timesteps"))
        epoch = self._first_number(metrics.get("train/epoch"), metrics.get("val/epoch"))
        t = time.time()

        row = {
            "time": t,
            # Respect configured precision for counters (precision=0 -> int)
            "total_timesteps": self._coerce_by_precision("total_timesteps", step),
            "epoch": self._coerce_by_precision("epoch", epoch),
        }
        for name, value in metrics.items():
            if not self._is_number(value):
                continue
            row[str(name)] = self._coerce_by_precision(str(name), value)

        # Nothing to write
        if len(row) == len(self._base_fields):
            return

        # Non-blocking put with drop-on-full to avoid stalling training
        try:
            self._q.put_nowait([row])
        except Full:
            # Queue full; drop this row to preserve throughput
            return

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
        # Main loop; wake periodically to check stop flag
        while not self._stop.is_set():
            self._drain_once(timeout=0.1)
        # Final drain on stop
        self._drain_once(timeout=0.0)

    def _drain_once(self, timeout: float) -> None:
        batched = []
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
        if not batched:
            return

        # Determine if header upgrade is needed
        current_set = set(self._fieldnames)
        # Collect any new metric keys excluding base and legacy fields
        new_keys = set()
        for d in batched:
            for k in d.keys():
                if k in self._base_fields or k in self._ignore_legacy_fields:
                    continue
                if k not in current_set:
                    new_keys.add(k)

        if new_keys:
            # Build new fieldnames: base + sorted(existing_metrics ∪ new_keys)
            existing_metrics = [f for f in self._fieldnames if f not in self._base_fields and f not in self._ignore_legacy_fields]
            merged = sorted(set(existing_metrics).union(new_keys))
            new_fieldnames = [*self._base_fields, *merged]
            self._rewrite_file_with_new_header(new_fieldnames, batched)
            return

        # Fast path: write rows with current header
        self._writer.writerows(batched)
        self._fh.flush()

    def _flush_remaining(self) -> None:
        # Write anything still in the queue
        remaining = []
        while True:
            try:
                remaining.extend(self._q.get_nowait())
            except Empty:
                break
        if remaining:
            # Ensure no header change needed unexpectedly
            current_set = set(self._fieldnames)
            extra = set()
            for d in remaining:
                for k in d.keys():
                    if k in self._base_fields or k in self._ignore_legacy_fields:
                        continue
                    if k not in current_set:
                        extra.add(k)
            if extra:
                existing_metrics = [f for f in self._fieldnames if f not in self._base_fields and f not in self._ignore_legacy_fields]
                merged = sorted(set(existing_metrics).union(extra))
                new_fieldnames = [*self._base_fields, *merged]
                self._rewrite_file_with_new_header(new_fieldnames, remaining)
            else:
                self._writer.writerows(remaining)
                self._fh.flush()

    @staticmethod
    def _is_number(x: Any) -> bool:
        return isinstance(x, numbers.Number)

    @staticmethod
    def _first_number(*values: Any) -> Optional[float]:
        for v in values:
            if isinstance(v, numbers.Number):
                return float(v)
        return None

    @staticmethod
    def _round_float(value: float, precision: int) -> float:
        try:
            return round(float(value), int(precision))
        except Exception:
            return float(value)

    def _coerce_by_precision(self, metric_key_or_bare: str, value: Any) -> Any:
        """Coerce a numeric metric to int when precision=0, else rounded float.

        Accepts either namespaced keys (e.g., "train/total_timesteps") or bare
        names (e.g., "total_timesteps"). Non-numeric or None values are returned
        as-is so the CSV writer can handle blanks appropriately.
        """
        if value is None or not self._is_number(value):
            return value

        try:
            precision = int(metrics_config.precision_for_metric(metric_key_or_bare))
        except Exception:
            precision = 2

        if precision == 0:
            # Cast to int to avoid trailing .0 in CSV (e.g., 3 instead of 3.0)
            try:
                return int(round(float(value)))
            except Exception:
                return int(float(value))
        else:
            # Keep as float but round to configured precision
            return self._round_float(float(value), precision)

    # -------- file/header helpers --------
    def _read_existing_header(self, path: Path) -> Optional[List[str]]:
        if not path.exists() or path.stat().st_size == 0:
            return None
        with open(path, mode="r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header and all(isinstance(c, str) for c in header):
                return [c.strip() for c in header]
        return None

    def _init_new_file_with_header(self, fieldnames: List[str]) -> None:
        self._fh = open(self.path, mode="w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()
        self._fh.flush()

    def _rotate_legacy_file(self) -> None:
        # Choose a non-clobbering legacy path
        base = self.path.with_suffix("")
        legacy = base.with_suffix(".legacy.csv")
        if legacy.exists():
            ts = time.strftime("%Y%m%d-%H%M%S")
            legacy = base.with_name(base.name + f".legacy-{ts}.csv")
        self.path.replace(legacy)

    def _rewrite_file_with_new_header(self, new_fieldnames: List[str], pending_rows: List[Dict[str, Any]]) -> None:
        """Atomically rewrite the CSV with an upgraded header.

        Reads existing rows (wide format) and writes them under the new header,
        then appends the pending rows. Finally, swaps the temp file in place.
        """
        # Close current handle to allow rename/replace
        self._fh.flush()
        self._fh.close()

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, mode="w", encoding="utf-8", newline="") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=new_fieldnames)
            writer.writeheader()

            # Copy old rows if any and if header was already wide
            old_header = self._read_existing_header(self.path)
            if old_header is not None and not ({"name", "value"}.issubset(set(old_header))):
                with open(self.path, mode="r", encoding="utf-8", newline="") as in_fh:
                    reader = csv.DictReader(in_fh)
                    for row in reader:
                        writer.writerow(row)

            # Append pending rows
            writer.writerows(pending_rows)

        # Atomic replace
        Path(tmp_path).replace(self.path)

        # Reopen for append with new header
        self._fieldnames = list(new_fieldnames)
        self._fh = open(self.path, mode="a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
