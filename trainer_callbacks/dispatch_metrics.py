from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

from utils.csv_logger import CsvMetricsLogger

class DispatchMetricsCallback(pl.Callback):

    def __init__(self, *, csv_path: str | Path, queue_size: int = 10000) -> None:
        super().__init__()
        self._path = Path(csv_path)
        self._csv_logger: Optional[CsvMetricsLogger] = None
        self._queue_size = int(queue_size)

    # ---- lifecycle hooks ----
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_logger = CsvMetricsLogger(self._path, queue_size=self._queue_size)

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.metrics.reset_epoch("train")  # type: ignore[attr-defined]
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Don't log until we have at least one episode completed 
        # (otherwise we won't be able to get reliable metrics)
        rollout_metrics = pl_module.train_collector.get_metrics()
        total_episodes = rollout_metrics.get("total_episodes", 0)
        if total_episodes == 0: return

        # Global & instant FPS from the same tracker
        total_timesteps = int(rollout_metrics["total_timesteps"])
        time_elapsed = pl_module._timing_tracker.seconds_since("on_fit_start")
        fps_total = pl_module._timing_tracker.fps_since("on_fit_start", steps_now=total_timesteps)
        fps_instant = pl_module._timing_tracker.fps_since("on_train_epoch_start", steps_now=total_timesteps)

        epoch_metrics = pl_module.metrics.compute_epoch_means("train")  # type: ignore[attr-defined]

        # Prepare metrics to log
        _metrics = {
            **{k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")},
            **epoch_metrics,
            "time_elapsed": time_elapsed,
            "epoch": pl_module.current_epoch,
            "fps": fps_total,
            "fps_instant": fps_instant,
        }

        # Derive ETA (seconds remaining) from FPS and max_timesteps if available
        if fps_total > 0.0 and pl_module.config.max_timesteps is not None:
            _metrics["eta_s"] = float(pl_module.config.max_timesteps / float(fps_total))

        prefixed_metrics = {f"train/{k}": v for k, v in _metrics.items()}

        # Write to CSV asynchronously and to Lightning for any UI consumers
        self._csv_logger.buffer_metrics(prefixed_metrics)
        
        pl_module.log_dict(prefixed_metrics)

        # Update step-aware history with aggregated snapshot
        pl_module.metrics.update_history(prefixed_metrics)  # type: ignore[attr-defined]
        
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not pl_module._should_run_eval(pl_module.current_epoch): return

        pl_module.metrics.reset_epoch("eval")  # type: ignore[attr-defined]
    

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not pl_module._should_run_eval(pl_module.current_epoch): return
        
        # Prepare metrics to log
        rollout_metrics = pl_module.validation_collector.get_metrics()

        epoch_metrics = pl_module.metrics.compute_epoch_means("eval")  # type: ignore[attr-defined]

        _metrics = {
            **{k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")},
            **epoch_metrics,
            "epoch": pl_module.current_epoch,
        }

        prefixed_metrics = {f"eval/{k}": v for k, v in _metrics.items()}

        self._csv_logger.buffer_metrics(prefixed_metrics)
        
        pl_module.log_dict(prefixed_metrics)

        pl_module.metrics.update_history(prefixed_metrics)  # type: ignore[attr-defined]
        
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._close()

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str | None = None) -> None:
        self._close()

    def _close(self) -> None:
        if self._csv_logger is None: return
        self._csv_logger.close()
        self._csv_logger = None
