from __future__ import annotations

import pytorch_lightning as pl

class DispatchMetricsCallback(pl.Callback):

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Reset epoch metrics
        pl_module.metrics.reset_epoch("train")
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Dispatch metrics
        self._dispatch_metrics(pl_module, "train")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not pl_module._should_run_eval(pl_module.current_epoch): return # TODO: should_run_validation_epoch()?

        # Reset epoch metrics
        pl_module.metrics.reset_epoch("eval")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not pl_module._should_run_eval(pl_module.current_epoch): return
        
        # Dispatch metrics
        self._dispatch_metrics(pl_module, "eval")
    
    def _dispatch_metrics(self, pl_module: pl.LightningModule, stage: str):
        # Don't log until we have at least one episode completed 
        # (otherwise we won't be able to get reliable metrics)
        rollout_collector = pl_module.get_rollout_collector(stage)
        rollout_metrics = rollout_collector.get_metrics()
        # TODO: review this, causes issues for "val" stage
        #total_episodes = rollout_metrics.get("total_episodes", 0)
        #if total_episodes == 0: return

        # Calculate timing metrics
        total_timesteps = int(rollout_metrics["total_timesteps"])
        time_elapsed = pl_module._timing_tracker.seconds_since("on_fit_start")
        fps_total = pl_module._timing_tracker.fps_since("on_fit_start", steps_now=total_timesteps)
        fps_instant = pl_module._timing_tracker.fps_since("on_train_epoch_start", steps_now=total_timesteps)

        # Aggregate metrics for the this epoch
        epoch_metrics = pl_module.metrics.compute_epoch_means(stage)

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
        pl_module.metrics.update_history(prefixed_metrics)