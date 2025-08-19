"""CSV metrics logger as a PyTorch Lightning Callback.

Moves CSV writing concerns out of BaseAgent and into a reusable callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pytorch_lightning as pl

from utils.csv_logger import CsvMetricsLogger


class CSVMetricsLoggerCallback(pl.Callback):
    """Flushes aggregated metrics from the LightningModule to a wide-form CSV.

    This callback expects the LightningModule to accumulate metrics in a
    `MetricsBuffer` and expose a `_flush_metrics()` method that returns a
    dict of aggregated means and clears the buffer, while also forwarding to
    Lightning's `log_dict` for standard logging.

    Parameters
    ----------
    csv_path : str | Path
        Destination CSV file path. Parent directories will be created.
    queue_size : int
        Max async queue size for the underlying CsvMetricsLogger.
    """

    def __init__(self, *, csv_path: str | Path, queue_size: int = 10000) -> None:
        super().__init__()
        self._path = Path(csv_path)
        self._logger: Optional[CsvMetricsLogger] = None
        self._queue_size = int(queue_size)

    # ---- lifecycle hooks ----
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = CsvMetricsLogger(self._path, queue_size=self._queue_size)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._flush_from_module(pl_module, allow_lightning_logging=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._flush_from_module(pl_module, allow_lightning_logging=True)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # on_fit_end does not allow LightningModule.log(); flush without forwarding to Lightning
        self._flush_from_module(pl_module, allow_lightning_logging=False)
        self._close()

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        self._close()

    # ---- internals ----
    def _flush_from_module(self, pl_module: pl.LightningModule, *, allow_lightning_logging: bool) -> None:
        # Use module's flush if available, otherwise do nothing.
        flush_fn = getattr(pl_module, "_flush_metrics", None)
        if flush_fn is None:
            return
        try:
            means: Dict[str, Any] = flush_fn(log_to_lightning=allow_lightning_logging) or {}
        except TypeError:
            # Backward-compat: older signature without return value
            flush_fn()
            means = {}
        if self._logger and means:
            self._logger.log_metrics(means)
        # If Lightning logging was not allowed here, attempt to forward to the
        # underlying experiment (e.g., W&B) to preserve the final metrics.
        if not allow_lightning_logging and means:
            try:
                exp = getattr(getattr(pl_module, "logger", None), "experiment", None)
                if exp is not None and hasattr(exp, "log"):
                    step = means.get("train/total_timesteps") or means.get("eval/total_timesteps")
                    if step is not None:
                        exp.log(means, step=int(step))
                    else:
                        exp.log(means)
            except Exception:
                # Never fail training due to logging issues
                pass

    def _close(self) -> None:
        try:
            if self._logger:
                self._logger.close()
        finally:
            self._logger = None
