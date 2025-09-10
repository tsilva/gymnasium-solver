from __future__ import annotations

import pytorch_lightning as pl

class AggregateMetricsCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module._epoch_metrics_buffer.clear()

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

        epoch_metrics = pl_module._epoch_metrics_buffer.means()

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

        prefixed = {f"train/{k}": v for k, v in _metrics.items()}
        
        # Store aggregated metrics for dispatchers to use
        pl_module._last_epoch_metrics = prefixed

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module._epoch_metrics_buffer.clear()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Prepare metrics to log
        rollout_metrics = pl_module.validation_collector.get_metrics()

        epoch_metrics = pl_module._epoch_metrics_buffer.means()

        _metrics = {
            **{k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")},
            **epoch_metrics,
            "epoch": pl_module.current_epoch,
        }

        prefixed = {f"eval/{k}": v for k, v in _metrics.items()}

        # Store aggregated metrics for dispatchers to use
        pl_module._last_epoch_metrics = prefixed
