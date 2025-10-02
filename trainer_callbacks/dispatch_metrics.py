from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import wandb


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

        # Skip logging if no episodes have completed yet (prevents -inf for ep_rew_best)
        if rollout_metrics.get("cnt/total_episodes", 0) == 0:
            return

        # Calculate timing metrics
        time_elapsed = pl_module.timings.seconds_since("on_fit_start")
        fps_total_map = pl_module.timings.throughput_since("on_fit_start", values_now=rollout_metrics)
        fps_instant_map = pl_module.timings.throughput_since("on_train_epoch_start", values_now=rollout_metrics)
        # Prefer vectorized-step FPS to align with step_key and early stopping
        fps_total = fps_total_map.get("cnt/total_timesteps", fps_total_map.get("roll/timesteps", 0.0))
        fps_instant = fps_instant_map.get("cnt/total_timesteps", fps_instant_map.get("roll/timesteps", 0.0)) # TODO: remove default

        # Aggregate metrics for the this epoch
        epoch_metrics = pl_module.metrics_recorder.compute_epoch_means(stage)

        # Extract action distribution for histogram logging before filtering
        action_dist = rollout_metrics.get("action_dist", None)

        # Discard distribution metrics (not loggable)
        filtered_rollout_metrics = {k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")}

        # Prepare metrics to log
        loggable_metrics = {
            **filtered_rollout_metrics,
            **epoch_metrics,
            "sys/timing/time_elapsed": time_elapsed,
            "cnt/epoch": pl_module.current_epoch,
            "sys/timing/fps": fps_total,
            "sys/timing/fps_instant": fps_instant,
        }

        # Add training progress metric when a max env steps budget is defined.
        # Progress is computed from env steps to align with the step key
        # and early stopping logic (0.0 at start → 1.0 at/after max_env_steps).
        if stage == "train" and getattr(pl_module.config, "max_env_steps", None) is not None:
            try:
                progress = float(pl_module.calc_training_progress())
            except Exception:
                progress = None
            if progress is not None:
                loggable_metrics["progress"] = progress

        # Derive ETA (seconds remaining) from vec-step FPS and max_env_steps if available
        if fps_total > 0.0 and pl_module.config.max_env_steps is not None:
            loggable_metrics["sys/timing/eta_s"] = float(pl_module.config.max_env_steps / float(fps_total))

        # Prefix metrics with train/
        prefixed_metrics = {f"{stage}/{k}": v for k, v in loggable_metrics.items()}

        # Flush metrics to Lightning (W&B, CSV logger, etc.)
        pl_module.log_dict(prefixed_metrics)

        # Log action distribution histogram to wandb separately (not sent to other loggers)
        if action_dist is not None and wandb.run is not None:
            # Create histogram from action counts
            # action_dist is an array where index i contains count for action i
            action_indices = []
            for action_idx, count in enumerate(action_dist):
                action_indices.extend([action_idx] * int(count))
            if len(action_indices) > 0:
                wandb.log({f"{stage}/roll/actions/histogram": wandb.Histogram(np_histogram=(action_dist, np.arange(len(action_dist) + 1)))})

        # Update step-aware history with aggregated snapshot
        pl_module.metrics_recorder.update_history(prefixed_metrics)
