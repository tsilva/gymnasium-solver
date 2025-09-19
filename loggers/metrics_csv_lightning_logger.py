from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pytorch_lightning.loggers.logger import (
    Logger as LightningLoggerBase,  # type: ignore
)

from loggers.metrics_csv_logger import MetricsCSVLogger

# TODO: REFACTOR this file

class MetricsCSVLightningLogger(LightningLoggerBase):
    """
    PyTorch Lightning logger that writes metrics to a wide-form CSV via
    loggers.metrics_csv_logger.MetricsCSVLogger. Accepts the same metrics dict that
    pl_module.log_dict emits; non-numeric values are ignored.
    """

    def __init__(self, *, csv_path: str | Path, queue_size: int = 10000) -> None:
        self._name = "csv"
        self._version = "0"
        self._experiment = None
        self._csv = MetricsCSVLogger(csv_path, queue_size=queue_size)

    # --- Lightning Logger API ---
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    @property
    def version(self) -> str | int:  # pragma: no cover - trivial
        return self._version

    def log_hyperparams(self, params: Any) -> None:  # pragma: no cover - unused
        # Hyperparams are already saved to config.json; no-op here
        return None

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        # Delegate to the async CSV writer; ignores non-numeric values internally
        self._csv.buffer_metrics(metrics)

    def finalize(self, status: str) -> None:  # pragma: no cover - best-effort close
        self._csv.close()

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:  # pragma: no cover - unused
        return None
