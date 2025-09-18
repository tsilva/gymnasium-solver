"""Checkpoint management utilities for model saving and resuming training."""

import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch

from utils.filesystem import update_symlink
from utils.io import write_json
from utils.scalars import only_scalar_values


class ModelCheckpointCallback(pl.Callback):
    """Custom checkpoint callback that handles all model checkpointing logic including resume."""

    def __init__(
        self,
        checkpoint_dir: str,
        metric: str,
        mode: str = "max",
    ):
        """Initialize the checkpoint callback.

        Args:
            checkpoint_dir: Base directory for checkpoints
            metric: Metric key to monitor in `trainer.logged_metrics`
            mode: 'max' or 'min' for the monitored metric
        """
        # Store attributes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric = metric
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int | None = None
        self.last_epoch: int | None = None  # we keep this attribute

        # Ensure the checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(
        self,
        agent: pl.LightningModule,
        epoch: int,
        metrics: dict[str, Any],
    ) -> tuple[Path, float]:
        """Save a checkpoint and return the path plus wall-clock save duration."""
        start_ns = time.perf_counter_ns()

        checkpoint_path = self.checkpoint_dir / f"epoch={epoch:02d}.ckpt"
        checkpoint_data = {"model_state_dict": agent.policy_model.state_dict()}
        torch.save(checkpoint_data, checkpoint_path)

        # Save the metrics snapshot at this epoch
        json_path = checkpoint_path.with_suffix(".json")
        write_json(json_path, metrics)

        duration_s = max((time.perf_counter_ns() - start_ns) / 1e9, 0.0)
        return checkpoint_path, duration_s

    def _sync_symlinks_for_epoch(self, epoch: int, tag: str) -> None:
        """Create/update symlinks like `<tag>.<ext>` for all files of a given epoch.

        Example:
            epoch=07.ckpt -> {tag}.ckpt
            epoch=07.json -> {tag}.json
            epoch=07.mp4  -> {tag}.mp4

        Args:
            epoch: Epoch number whose artifacts should be linked.
            tag: Prefix for symlinks (e.g., 'last' or 'best').
        """
        # Match any file starting with the epoch prefix. This covers .ckpt, .json, .mp4, etc.
        prefix = f"epoch={epoch:02d}"
        for f in self.checkpoint_dir.glob(f"{prefix}*"):
            # Skip non-files
            if not f.is_file(): continue

            # Preserve full multi-suffix (e.g., '.tar.gz' if that ever occurs)
            suffix = "".join(f.suffixes)
            link_path = self.checkpoint_dir / f"{tag}{suffix}"
            update_symlink(link_path, f)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save checkpoints on eval epochs and handle best/last/threshold logic.

        During warmup epochs (or when eval is skipped), the monitored metric may
        not be present in `trainer.logged_metrics`. In that case, gracefully
        return without asserting, keeping early-stopping/tracking logic intact.
        """
        # If the monitored metric wasn't logged this epoch (e.g., warmup), skip
        if self.metric not in trainer.logged_metrics:
            return

        # Always save a checkpoint for this eval epoch
        epoch = pl_module.current_epoch
        metrics = only_scalar_values(trainer.logged_metrics)
        _, save_duration_s = self._save_checkpoint(pl_module, epoch, metrics)

        # Track "last" epoch seen with a checkpoint
        self.last_epoch = epoch

        # Log checkpoint timing metrics to the unified logging stream (e.g., W&B)
        checkpoint_metrics: dict[str, float] = {"checkpoint/save_duration_s": save_duration_s}

        timings = getattr(pl_module, "timings", None)
        if timings is not None:
            try:
                elapsed = float(timings.seconds_since("on_fit_start"))
            except (KeyError, AttributeError):
                elapsed = None
            if elapsed is not None:
                checkpoint_metrics["checkpoint/time_elapsed_s"] = elapsed

        if checkpoint_metrics:
            recorder = getattr(pl_module, "metrics_recorder", None)
            if recorder is not None:
                recorder.record("val", checkpoint_metrics)

            pl_module.log_dict(
                {f"val/{k}": v for k, v in checkpoint_metrics.items()},
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        # Determine if this is the best checkpoint so far
        metric_value = trainer.logged_metrics[self.metric]
        is_best = (metric_value > self.best_value) if (self.mode == "max") else (metric_value < self.best_value)

        # If this is the best checkpoint so far, update the best values
        if is_best:
            self.best_value = float(metric_value)
            self.best_epoch = epoch

            # Record the best checkpoint metadata without printing from the callback layer.
            best_metrics = {
                "checkpoint/best_epoch": float(epoch),
                "checkpoint/best_metric_value": float(metric_value),
            }
            recorder = getattr(pl_module, "metrics_recorder", None)
            if recorder is not None:
                recorder.record("val", best_metrics)

            pl_module.log_dict({f"val/{k}": v for k, v in best_metrics.items()}, logger=True, on_step=False, on_epoch=True)

        """Handle training completion summary and ensure symlinks are consistent."""
        # Ensure `last.*` symlinks reflect the final checkpointed epoch
        if self.last_epoch is not None: self._sync_symlinks_for_epoch(self.last_epoch, tag="last")

        # Ensure `best.*` symlinks reflect the best checkpointed epoch
        if self.best_epoch is not None: self._sync_symlinks_for_epoch(self.best_epoch, tag="best")
