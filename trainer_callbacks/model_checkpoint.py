"""Checkpoint management utilities for model saving and resuming training."""

from pathlib import Path
import shutil
import os
import torch

from utils.io import write_json

import pytorch_lightning as pl
import numpy as np
import numbers
from typing import Any, Dict

def _to_scalar(x: Any) -> Any:
    """Return a Python scalar (int/float/bool) if x is scalar-like, else None."""
    # Built-in numeric types are fine
    if isinstance(x, numbers.Number) or isinstance(x, bool):
        return x

    # NumPy scalar (e.g., np.float32(3.14)) -> Python scalar
    if isinstance(x, np.generic):
        return x.item()

    # NumPy array with exactly one element
    if isinstance(x, np.ndarray):
        if x.ndim == 0 or x.size == 1:
            return x.reshape(()).item()
        return None  # non-scalar -> skip

    # PyTorch tensor with exactly one element
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return None  # non-scalar -> skip

    # Anything else: try a last-resort cast to float if it clearly acts scalar
    # (avoid strings/containers). If it fails, skip.
    try:
        return float(x)
    except Exception:
        return None

def _only_scalars(d: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    skipped = []
    for k, v in d.items():
        sv = _to_scalar(v)
        if sv is None:
            skipped.append(k)
        else:
            cleaned[k] = sv
    return cleaned


# TODO: move to reusable util
def update_symlink(link_path: Path, target_path: Path) -> None:
    """Create or update a symlink at link_path pointing to target_path.

    Uses a relative target path from the link's parent directory for portability.
    Ensures parent directories exist. Falls back to copying on platforms that
    do not support symlinks or when permissions are insufficient.
    """
    link_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing link/file if present
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
    except FileNotFoundError:
        pass

    # Compute relative target from link directory for robustness
    rel_target = os.path.relpath(str(target_path), start=str(link_path.parent))

    try:
        link_path.symlink_to(rel_target)
    except (NotImplementedError, OSError, PermissionError):
        # Symlinks not supported or blocked; copy file as a fallback.
        # Ensure we copy the file contents rather than linking.
        shutil.copy2(target_path, link_path)
        

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
    ) -> Path:
        """Save a checkpoint with all necessary information, including a metrics snapshot."""
        checkpoint_path = self.checkpoint_dir / f"epoch={epoch:02d}.ckpt"
        checkpoint_data = {"model_state_dict": agent.policy_model.state_dict()}
        torch.save(checkpoint_data, checkpoint_path)

        # Save the metrics snapshot at this epoch
        json_path = checkpoint_path.with_suffix(".json")    
        write_json(json_path, metrics)

        return checkpoint_path

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
        # If eval shouldn't be run this epoch, skip the step (e.g., warmup epochs)
        if not pl_module.should_run_validation_epoch():
            return

        # If the monitored metric wasn't logged this epoch (e.g., warmup), skip
        if self.metric not in trainer.logged_metrics:
            return

        # Always save a checkpoint for this eval epoch
        epoch = pl_module.current_epoch
        metrics = _only_scalars(trainer.logged_metrics)
        self._save_checkpoint(pl_module, epoch, metrics)

        # Track "last" epoch seen with a checkpoint
        self.last_epoch = epoch

        # Determine if this is the best checkpoint so far
        metric_value = trainer.logged_metrics[self.metric]
        is_best = (metric_value > self.best_value) if (self.mode == "max") else (metric_value < self.best_value)

        # If this is the best checkpoint so far, update the best values
        if is_best:
            self.best_value = float(metric_value)
            self.best_epoch = epoch
            print(f"New best model: epoch={epoch}; {self.metric}={float(metric_value):.4f}")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Handle training completion summary and ensure symlinks are consistent."""
        # Ensure `last.*` symlinks reflect the final checkpointed epoch
        if self.last_epoch is not None: self._sync_symlinks_for_epoch(self.last_epoch, tag="last")

        # Ensure `best.*` symlinks reflect the best checkpointed epoch
        if self.best_epoch is not None: self._sync_symlinks_for_epoch(self.best_epoch, tag="best")