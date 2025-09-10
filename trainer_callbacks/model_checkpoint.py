"""Checkpoint management utilities for model saving and resuming training."""

from pathlib import Path
import json
import numpy as np
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
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric = metric
        self.mode = mode
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = None
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        checkpoint_path: Path, 
        *, 
        metrics: dict | None = None, 
        current_eval_reward: float | None = None
    ):
        """Save a checkpoint with all necessary information, including metrics snapshot,
        and the current evaluation reward.

        Args:   
            agent: LightningModule/Agent instance
            checkpoint_path: destination file path
            is_best: flag if this is best checkpoint
            is_last: flag if this is last checkpoint
            metrics: optional metrics dict (already serialized to JSON-serializable types)
            current_eval_reward: optional explicit eval reward value to store
        """
        from dataclasses import asdict, is_dataclass
        
        # Helper to serialize metric values to plain Python types
        def _to_py(val):
            import torch
            if isinstance(val, torch.Tensor):
                if val.ndim == 0:
                    return val.item()
                return val.detach().cpu().tolist()
            if isinstance(val, (np.floating, np.integer)):
                return val.item()
            # Plain numbers or strings pass through; others fallback to str
            if isinstance(val, (int, float, bool)):
                return val
            return str(val)

        def _serialize_metrics(m):
            if m is None:
                return None
            return {str(k): _to_py(v) for k, v in dict(m).items()}


        checkpoint_data = {
            'model_state_dict': agent.policy_model.state_dict(),
            'optimizer_state_dict': agent.optimizers().state_dict() if hasattr(agent.optimizers(), 'state_dict') else None,
            'config_dict': asdict(agent.config),
            'epoch': agent.current_epoch,
            'global_step': agent.global_step,
            'total_timesteps': getattr(agent, 'total_timesteps', 0),
            'best_eval_reward': getattr(agent, 'best_eval_reward', float('-inf')),
            'current_eval_reward': current_eval_reward,
            'metrics': _serialize_metrics(metrics),
            'rng_states': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Save metrics snapshot as a sidecar JSON with the same basename
        serialized_metrics = _serialize_metrics(metrics)
        if serialized_metrics is not None:
            json_path = checkpoint_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serialized_metrics, f, ensure_ascii=False, indent=2)
        return checkpoint_path
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Save checkpoints on eval epochs and handle best/last/threshold logic.

        During warmup epochs (or when eval is skipped), the monitored metric may
        not be present in `trainer.logged_metrics`. In that case, gracefully
        return without asserting, keeping early-stopping/tracking logic intact.
        """

        if not pl_module.should_run_validation_epoch(): return

        # If the monitored metric wasn't logged this epoch (e.g., warmup), skip
        if self.metric not in trainer.logged_metrics:
            self._handle_early_stopping_and_tracking(trainer, pl_module)
            return

        # Convert to float defensively
        current_metric_value = trainer.logged_metrics[self.metric]

        # Always save a timestamped checkpoint for this eval epoch
        epoch = pl_module.current_epoch
        timestamped_path = self.checkpoint_dir / f"epoch={epoch:02d}.ckpt"

        """Check if current metric value is better than the best seen so far."""
        is_best = False   
        if self.mode == 'max': is_best = current_metric_value > self.best_value
        else: is_best = current_metric_value < self.best_value

        # Save timestamped checkpoint marked with flags
        self._save_checkpoint(
            pl_module,
            timestamped_path,
            is_best=is_best,
            metrics=trainer.logged_metrics,
            current_eval_reward=current_metric_value,
        )

        # If this is a new best, update best metric and best symlinks
        if is_best:
            self.best_value = current_metric_value
            self.best_epoch = epoch
            print(f"New best model: {self.metric}={current_metric_value:.4f}\n  -> {timestamped_path.name}")

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
