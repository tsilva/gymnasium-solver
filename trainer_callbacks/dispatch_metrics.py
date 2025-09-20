from __future__ import annotations

import pytorch_lightning as pl


class DispatchMetricsCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Reset epoch metrics
        pl_module.metrics_recorder.reset_epoch("train")
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Dispatch metrics
        self._dispatch_metrics(pl_module, "train")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Reset epoch metrics
        pl_module.metrics_recorder.reset_epoch("val")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Dispatch metrics
        self._dispatch_metrics(pl_module, "val")

    def _dispatch_metrics(self, pl_module: pl.LightningModule, stage: str):
        # Don't log until we have at least one episode completed 
        # (otherwise we won't be able to get reliable metrics)
        rollout_collector = pl_module.get_rollout_collector(stage)
        rollout_metrics = rollout_collector.get_metrics()
        # TODO: review this, causes issues for "val" stage

        # Calculate timing metrics
        time_elapsed = pl_module.timings.seconds_since("on_fit_start")
        fps_total_map = pl_module.timings.throughput_since("on_fit_start", values_now=rollout_metrics)
        fps_instant_map = pl_module.timings.throughput_since("on_train_epoch_start", values_now=rollout_metrics)
        fps_total = fps_total_map["total_timesteps"]
        fps_instant = fps_instant_map["total_timesteps"]

        # Aggregate metrics for the this epoch
        epoch_metrics = pl_module.metrics_recorder.compute_epoch_means(stage)

        # Discard distribution metrics (not loggable)
        filtered_rollout_metrics = {k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")}

        # Prepare metrics to log
        loggable_metrics = {
            **filtered_rollout_metrics,
            **epoch_metrics,
            "time_elapsed": time_elapsed,
            "epoch": pl_module.current_epoch,
            "fps": fps_total,
            "fps_instant": fps_instant,
        }

        # Derive ETA (seconds remaining) from FPS and max_timesteps if available
        if fps_total > 0.0 and pl_module.config.max_timesteps is not None:
            loggable_metrics["eta_s"] = float(pl_module.config.max_timesteps / float(fps_total))

        # Prefix metrics with train/
        prefixed_metrics = {f"{stage}/{k}": v for k, v in loggable_metrics.items()}

        # Flush metrics to Lightning (W&B, CSV logger, etc.)
        pl_module.log_dict(prefixed_metrics)

        # Update step-aware history with aggregated snapshot
        pl_module.metrics_recorder.update_history(prefixed_metrics)
