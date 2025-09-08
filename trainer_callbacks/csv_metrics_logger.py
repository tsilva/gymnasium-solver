"""CSV metrics logger as a PyTorch Lightning Callback.

Moves CSV writing concerns out of BaseAgent and into a reusable callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
        metrics = pl_module._epoch_metrics_buffer.means()
        self._logger.log_metrics(metrics)   

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = pl_module._epoch_metrics_buffer.means()
        self._logger.log_metrics(metrics)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = pl_module._epoch_metrics_buffer.means()
        self._logger.log_metrics(metrics)
        self._close()

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        self._close()

    def _close(self) -> None:
        if self._logger is None: return
        self._logger.close()
        self._logger = None
