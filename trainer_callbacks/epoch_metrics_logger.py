from __future__ import annotations

import pytorch_lightning as pl

class EpochMetricsLoggerCallback(pl.Callback):

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

        # Prepare metrics to log
        _metrics = {
            **{k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")},
            "time_elapsed": time_elapsed,
            "epoch": pl_module.current_epoch,
            "fps": fps_total,
            "fps_instant": fps_instant,
        }

        # Derive ETA (seconds remaining) from FPS and max_timesteps if available
        if fps_total > 0.0 and pl_module.config.max_timesteps is not None:
            _metrics["eta_s"] = float(pl_module.config.max_timesteps / float(fps_total))
    
        # Log metrics to the buffer
        pl_module.log_metrics(_metrics, prefix="train")
