"""Checkpoint management utilities for model saving and resuming training."""

from pathlib import Path
import json
import torch

import pytorch_lightning as pl

class ModelCheckpointCallback(pl.Callback):
    """Custom checkpoint callback that handles all model checkpointing logic including resume."""
    
    def __init__(self, 
        checkpoint_dir: str,
        metric: str,
        mode: str = "max",
    ):
        """Initialize the checkpoint callback.

        Args:
            checkpoint_dir: Base directory for checkpoints
            monitor: Metric to monitor for saving best checkpoints
            mode: 'max' or 'min' for the monitored metric
            save_last: Whether to save the last checkpoint
        """

        # Store atributes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric = metric
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = None

        # Ensure the checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TODO: move to resuable util
    @staticmethod
    def _update_symlink(link_path: Path, target_path: Path) -> None:
        """Create or update a symlink at link_path pointing to target_path.

        Uses a relative target path from the link's parent directory for portability.
        Ensures parent directories exist. Falls back to copying on platforms that
        do not support symlinks.
        """
        # Ensure the link directory exists
        link_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing link/file if present
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        # Compute relative target from link directory for robustness
        import os
        rel_target = os.path.relpath(str(target_path), start=str(link_path.parent))

        link_path.symlink_to(rel_target)
    
    def _save_checkpoint(
        self, 
        agent: pl.LightningModule, 
        epoch: int, 
        *, 
        metrics: dict | None = None, 
    ):
        """Save a checkpoint with all necessary information, including metrics snapshot."""

        # Save the model checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch={epoch:02d}.ckpt"
        checkpoint_data = {'model_state_dict': agent.policy_model.state_dict()}
        torch.save(checkpoint_data, checkpoint_path)

        # TODO: use io util
        # Save the metrics snapshot at this epoch
        json_path = checkpoint_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f: json.dump(metrics, f, ensure_ascii=False, indent=2)
        return checkpoint_path
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save checkpoints on eval epochs and handle best/last/threshold logic.

        During warmup epochs (or when eval is skipped), the monitored metric may
        not be present in `trainer.logged_metrics`. In that case, gracefully
        return without asserting, keeping early-stopping/tracking logic intact.
        """

        # If eval shouldn't be run this epoch, skip the step (eg: warmup epochs)
        if not pl_module.should_run_validation_epoch(): return

        # If the monitored metric wasn't logged this epoch (e.g., warmup), skip
        if self.metric not in trainer.logged_metrics: return

        # Always save a timestamped checkpoint for this eval epoch
        epoch = pl_module.current_epoch
        self._save_checkpoint(
            pl_module,
            epoch,
            metrics=trainer.logged_metrics,
        )

        # If not best, return
        is_best = False   
        metric_value = trainer.logged_metrics[self.metric]
        if self.mode == 'max': is_best = metric_value > self.best_value
        else: is_best = metric_value < self.best_value
        if not is_best: return

        # Update best values
        self.best_value = metric_value
        self.best_epoch = epoch
        print(f"New best model: epoch={epoch}; {self.metric}={metric_value:.4f}")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Handle training completion summary and ensure symlinks are consistent."""

        if self.last_epoch is not None:
            last_epoch_ckpt = self.checkpoint_dir / f"epoch={self.last_epoch:02d}.ckpt"
            self._update_symlink(self.checkpoint_dir / "last.ckpt", last_epoch_ckpt)
            jpath = last_epoch_ckpt.with_suffix(".json")
            if jpath.exists(): self._update_symlink(self.checkpoint_dir / "last.json", jpath)
            vid = last_epoch_ckpt.with_suffix(".mp4")
            if vid.exists(): self._update_symlink(self.checkpoint_dir / "last.mp4", vid)

        # If we tracked a best epoch, ensure best symlinks exist
        if self.best_epoch is not None:
            last_epoch_ckpt = self.checkpoint_dir / f"epoch={self.best_epoch:02d}.ckpt"
            self._update_symlink(self.checkpoint_dir / "best.ckpt", last_epoch_ckpt)
            best_json = last_epoch_ckpt.with_suffix(".json")
            if best_json.exists(): self._update_symlink(self.checkpoint_dir / "best.json", best_json)
            best_vid = last_epoch_ckpt.with_suffix(".mp4")
            if best_vid.exists(): self._update_symlink(self.checkpoint_dir / "best.mp4", best_vid)
