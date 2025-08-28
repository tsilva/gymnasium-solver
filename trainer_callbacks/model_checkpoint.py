"""Checkpoint management utilities for model saving and resuming training."""

from pathlib import Path
import json

# Optional dependency shim for pytorch_lightning.Callback
try:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    BaseCallback = getattr(pl, "Callback", object)
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    BaseCallback = object

import torch


class ModelCheckpointCallback(BaseCallback):
    """Custom checkpoint callback that handles all model checkpointing logic including resume."""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 monitor: str = "eval/ep_rew_mean",
                 mode: str = "max",
                 save_last: bool = False,
                 save_threshold_reached: bool = True,
                 resume: bool = False):
        """Initialize the checkpoint callback.

        Args:
            checkpoint_dir: Base directory for checkpoints
            monitor: Metric to monitor for saving best checkpoints
            mode: 'max' or 'min' for the monitored metric
            save_last: Whether to save the last checkpoint
            save_threshold_reached: Whether to save when reward threshold is reached
            resume: Whether to attempt to resume from existing checkpoint
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_threshold_reached = save_threshold_reached
        self.resume = resume
        self._threshold_checkpoint_saved = False  # avoid duplicate threshold saves
        
        self.best_metric_value = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None
        self.resume_checkpoint_path = None
        self.best_epoch_index = None
        
    def setup(self, trainer, pl_module, stage):
        """Setup callback - called before training starts."""
        # Initialize best model tracking attributes on the agent for compatibility
        if not hasattr(pl_module, 'best_eval_reward'):
            pl_module.best_eval_reward = float('-inf')
        if not hasattr(pl_module, 'best_model_path'):
            pl_module.best_model_path = None
            
        if stage == "fit" and self.resume:
            self._handle_resume(pl_module)
    
    def _handle_resume(self, agent):
        """Handle resume logic - find and load checkpoint if available."""
        if getattr(agent.config, 'resume', False):
            from utils.checkpoint import find_latest_checkpoint, load_checkpoint
            checkpoint_path = find_latest_checkpoint(
                agent.config.algo_id, 
                agent.config.env_id, 
                getattr(agent.config, 'checkpoint_dir', 'checkpoints')
            )
            if checkpoint_path:
                self.resume_checkpoint_path = checkpoint_path
                print(f"Found checkpoint to resume from: {checkpoint_path}")
                # Load checkpoint after models are created
                load_checkpoint(checkpoint_path, agent, resume_training=True)
            else:
                print(f"Resume requested but no checkpoint found for {agent.config.algo_id}/{agent.config.env_id}")
    
    def _get_checkpoint_dir(self, agent) -> Path:
        """Get the checkpoint directory for this specific agent/env combination."""
        # If checkpoint_dir is already a run-specific directory, use it directly
        checkpoint_path = Path(self.checkpoint_dir)
        
        # Ensure the directory exists
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        return checkpoint_path

    @staticmethod
    def _update_symlink(link_path: Path, target_path: Path) -> None:
        """Create or update a symlink at link_path pointing to target_path.

        Uses a relative target path from the link's parent directory for portability.
        Ensures parent directories exist. Falls back to copying on platforms that
        do not support symlinks.
        """
        try:
            # Ensure the link directory exists
            link_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing link/file if present
            if link_path.exists() or link_path.is_symlink():
                try:
                    link_path.unlink()
                except Exception:
                    pass

            # Compute relative target from link directory for robustness
            try:
                import os
                rel_target = os.path.relpath(str(target_path), start=str(link_path.parent))
            except Exception:
                # Fallback to basename if relpath fails
                rel_target = target_path.name

            try:
                link_path.symlink_to(rel_target)
            except Exception as e:
                # As a last resort, copy the file (maintains correctness over portability)
                try:
                    import shutil
                    shutil.copy2(str(target_path), str(link_path))
                    print(f"Info: symlink unsupported; copied {target_path} -> {link_path} ({e})")
                except Exception as e2:
                    print(f"Warning: failed to create symlink or copy {link_path} -> {target_path}: {e2}")
        except Exception as e:
            print(f"Warning: failed to create symlink {link_path} -> {target_path}: {e}")

    @staticmethod
    def _find_latest_epoch_ckpt(checkpoint_dir: Path) -> Path | None:
        files = list(checkpoint_dir.glob("epoch=*.ckpt"))
        if not files:
            return None
        files.sort(key=lambda p: p.stat().st_mtime)
        return files[-1]
    
    def _save_checkpoint(self, agent, checkpoint_path: Path, is_best: bool = False, is_last: bool = False, is_threshold: bool = False, *, metrics: dict | None = None, current_eval_reward: float | None = None, threshold_value: float | None = None):
        """Save a checkpoint with all necessary information, including metrics snapshot.

        Args:
            agent: LightningModule/Agent instance
            checkpoint_path: destination file path
            is_best: flag if this is best checkpoint
            is_last: flag if this is last checkpoint
            is_threshold: flag if threshold reached checkpoint
            metrics: optional metrics dict (already serialized to JSON-serializable types)
            current_eval_reward: optional explicit eval reward value to store
            threshold_value: optional threshold used when saving
        """
        from dataclasses import asdict, is_dataclass
        
        # Helper to serialize metric values to plain Python types
        def _to_py(val):
            try:
                import torch
                import numpy as np
                if isinstance(val, torch.Tensor):
                    if val.ndim == 0:
                        return val.item()
                    return val.detach().cpu().tolist()
                if isinstance(val, (np.floating, np.integer)):
                    return val.item()
            except Exception:
                pass
            # Plain numbers or strings pass through; others fallback to str
            if isinstance(val, (int, float, bool)):
                return val
            return str(val)

        def _serialize_metrics(m):
            if m is None:
                return None
            try:
                return {str(k): _to_py(v) for k, v in dict(m).items()}
            except Exception:
                # Best-effort fallback
                try:
                    return {str(k): str(v) for k, v in dict(m).items()}
                except Exception:
                    return None

        # Prepare checkpoint data
        # Serialize config robustly (works with dataclass and simple objects)
        try:
            if is_dataclass(agent.config):
                cfg_dict = asdict(agent.config)
            elif isinstance(agent.config, dict):
                cfg_dict = {str(k): _to_py(v) for k, v in dict(agent.config).items()}
            else:
                try:
                    cfg_dict = {str(k): _to_py(v) for k, v in vars(agent.config).items()}
                except Exception:
                    cfg_dict = {"repr": str(agent.config)}
        except Exception:
            cfg_dict = None
        checkpoint_data = {
            'model_state_dict': agent.policy_model.state_dict(),
            'optimizer_state_dict': agent.optimizers().state_dict() if hasattr(agent.optimizers(), 'state_dict') else None,
            'config_dict': cfg_dict,
            'epoch': agent.current_epoch,
            'global_step': agent.global_step,
            'total_timesteps': getattr(agent, 'total_timesteps', 0),
            'best_eval_reward': getattr(agent, 'best_eval_reward', float('-inf')),
            'current_eval_reward': current_eval_reward,
            'is_best': is_best,
            'is_last': is_last,
            'is_threshold': is_threshold,
            'threshold_value': threshold_value,
            'metrics': _serialize_metrics(metrics),
            'rng_states': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Save metrics snapshot as a sidecar JSON with the same basename
        try:
            serialized_metrics = _serialize_metrics(metrics)
            if serialized_metrics is not None:
                json_path = checkpoint_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(serialized_metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Don't fail training if metrics JSON can't be written
            print(f"Warning: failed to write metrics JSON for {checkpoint_path.name}: {e}")
        return checkpoint_path
    
    def _should_save_best(self, current_value: float) -> bool:
        """Check if current metric value is better than the best seen so far."""
        if self.mode == 'max':
            return current_value > self.best_metric_value
        else:
            return current_value < self.best_metric_value
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save a timestamped checkpoint on eval epochs; skip cleanly when eval is gated (warmup)."""

        # During warmup, validation loop still advances but eval hooks may skip logging.
        # Be tolerant: if the monitored metric isn't present, treat this as a non-eval epoch.
        logged_metrics = getattr(trainer, "logged_metrics", {})

        current_metric_value = None
        if self.monitor in logged_metrics:
            try:
                current_metric_value = float(logged_metrics[self.monitor])
            except Exception:
                current_metric_value = None

        if current_metric_value is None:
            # No eval metrics this epoch (likely warmup); nothing to checkpoint.
            # Still allow early stopping/tracking logic to run gracefully.
            self._handle_early_stopping_and_tracking(trainer, pl_module)
            return

        checkpoint_dir = self._get_checkpoint_dir(pl_module)

        # Always save a timestamped checkpoint for this eval epoch
        epoch = pl_module.current_epoch
        step = pl_module.global_step
        timestamped_path = checkpoint_dir / f"epoch={epoch:02d}.ckpt"

        # Determine reward threshold once for this epoch
        cfg_thr = getattr(pl_module.config, 'reward_threshold', None)
        effective_threshold = cfg_thr
        if effective_threshold is None:
            try:
                effective_threshold = pl_module.validation_env.get_reward_threshold()
            except Exception:
                effective_threshold = None
        if effective_threshold is None:
            try:
                import gymnasium as gym
                spec = gym.spec(pl_module.config.env_id)
                if spec and hasattr(spec, 'reward_threshold'):
                    effective_threshold = getattr(spec, 'reward_threshold')
            except Exception:
                effective_threshold = None

        is_best = self._should_save_best(current_metric_value)
        is_threshold = (
            self.save_threshold_reached and
            effective_threshold is not None and
            current_metric_value >= effective_threshold and
            not self._threshold_checkpoint_saved
        )

        # Save timestamped checkpoint marked with flags
        self._save_checkpoint(
            pl_module,
            timestamped_path,
            is_best=is_best,
            is_threshold=is_threshold,
            metrics=logged_metrics,
            current_eval_reward=current_metric_value,
            threshold_value=effective_threshold,
        )

        # Maintain 'last' symlinks (ckpt + json + mp4) to this epoch artifacts
        last_ckpt = checkpoint_dir / "last.ckpt"
        self._update_symlink(last_ckpt, timestamped_path)
        self.last_checkpoint_path = str(last_ckpt)

        # Update last.json -> epoch=XX.json when available
        epoch_json = checkpoint_dir / f"epoch={epoch:02d}.json"
        if epoch_json.exists():
            self._update_symlink(checkpoint_dir / "last.json", epoch_json)

        # Update last.mp4 -> epoch=XX.mp4 when available
        epoch_video = checkpoint_dir / f"epoch={epoch:02d}.mp4"
        if epoch_video.exists():
            self._update_symlink(checkpoint_dir / "last.mp4", epoch_video)

        # If this is a new best, update best metric and best symlinks
        if is_best:
            self.best_metric_value = current_metric_value
            self.best_epoch_index = int(epoch)
            best_ckpt = checkpoint_dir / "best.ckpt"
            self._update_symlink(best_ckpt, timestamped_path)
            self.best_checkpoint_path = str(best_ckpt)

            # Mirror best.json if available
            if epoch_json.exists():
                self._update_symlink(checkpoint_dir / "best.json", epoch_json)

            if epoch_video.exists():
                self._update_symlink(checkpoint_dir / "best.mp4", epoch_video)

            print(f"New best model: {self.monitor}={current_metric_value:.4f}\n  -> {timestamped_path.name}")
            pl_module.best_model_path = str(best_ckpt)
            pl_module.best_eval_reward = current_metric_value

        # Unified reward threshold (config overrides env spec, then gym.spec fallback)
        if is_threshold:
            print(
                f"Threshold reached! Saved model with {self.monitor}={current_metric_value:.4f} "
                f"(threshold={effective_threshold}) at {timestamped_path}"
            )
            self._threshold_checkpoint_saved = True
            if not trainer.should_stop and getattr(pl_module.config, 'early_stop_on_eval_threshold', True):
                print(
                    f"Early stopping at epoch {pl_module.current_epoch} with eval mean reward {current_metric_value:.2f} >= threshold {effective_threshold}"
                )
                trainer.should_stop = True

        # Delegate to legacy tracking (will early stop if missed above)
        self._handle_early_stopping_and_tracking(trainer, pl_module)
    
    def _handle_early_stopping_and_tracking(self, trainer, pl_module):
        """Handle early stopping logic and best reward tracking that was previously in BaseAgent."""
        # Get reward threshold - use config if provided, otherwise use environment's reward threshold
        config_threshold = pl_module.config.reward_threshold
        if config_threshold is not None: 
            reward_threshold = config_threshold
            print(f"Using config reward_threshold: {reward_threshold}")
        else: 
            # Attempt retrieval through wrapper
            try:
                reward_threshold = pl_module.validation_env.get_reward_threshold()
            except Exception:
                reward_threshold = None
            if reward_threshold is None:
                # Fallback to direct gym spec lookup
                try:
                    import gymnasium as gym
                    spec = gym.spec(pl_module.config.env_id)
                    if spec and hasattr(spec, 'reward_threshold'):
                        reward_threshold = getattr(spec, 'reward_threshold')
                        if reward_threshold is not None:
                            print(f"Using gym.spec reward_threshold fallback: {reward_threshold}")
                except Exception:
                    reward_threshold = None

        # Get current eval metrics
        if hasattr(trainer, 'logged_metrics') and "eval/ep_rew_mean" in trainer.logged_metrics:
            ep_rew_mean = float(trainer.logged_metrics["eval/ep_rew_mean"])
            
            # Check for early stopping based on reward threshold (only if threshold is available)
            if reward_threshold is not None:
                # Update best_eval_reward for compatibility and early stopping logic
                if ep_rew_mean > pl_module.best_eval_reward:
                    pl_module.best_eval_reward = ep_rew_mean
                
                # Note: Early stopping for threshold is already handled above in checkpoint saving
                # This is just for the case where threshold stopping wasn't triggered by checkpoint saving
                if ep_rew_mean >= reward_threshold and not trainer.should_stop and getattr(pl_module.config, 'early_stop_on_eval_threshold', True):
                    print(f"Early stopping at epoch {pl_module.current_epoch} with eval mean reward {ep_rew_mean:.2f} >= threshold {reward_threshold}")
                    trainer.should_stop = True
            else:
                # Even without reward threshold, update best eval reward for tracking
                if ep_rew_mean > pl_module.best_eval_reward:
                    pl_module.best_eval_reward = ep_rew_mean
                
                # Provide more informative logging about why reward threshold is not available
                env_threshold = pl_module.validation_env.get_reward_threshold()
                print(f"No reward threshold available - config: {config_threshold}, env spec: {env_threshold} - skipping early stopping check")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Maintain 'last' symlinks to the most recent epoch checkpoint/video after each training epoch."""
        # Train-side early stopping on threshold if enabled
        try:
            train_ep_rew_mean = float(trainer.logged_metrics.get("train/ep_rew_mean")) if hasattr(trainer, 'logged_metrics') else None
        except Exception:
            train_ep_rew_mean = None

        # Determine threshold using config first, then env/gym if needed
        threshold = getattr(pl_module.config, 'reward_threshold', None)
        if threshold is None:
            try:
                threshold = pl_module.train_env.get_reward_threshold()
            except Exception:
                threshold = None
        if threshold is None:
            try:
                import gymnasium as gym
                spec = gym.spec(pl_module.config.env_id)
                if spec and hasattr(spec, 'reward_threshold'):
                    threshold = getattr(spec, 'reward_threshold')
            except Exception:
                threshold = None

        if (getattr(pl_module.config, 'early_stop_on_train_threshold', False) and
            threshold is not None and
            train_ep_rew_mean is not None and
            not getattr(trainer, 'should_stop', False) and
            train_ep_rew_mean >= float(threshold)):
            print(f"Early stopping at epoch {pl_module.current_epoch} with train mean reward {train_ep_rew_mean:.2f} >= threshold {threshold}")
            trainer.should_stop = True

        if self.save_last:
            checkpoint_dir = self._get_checkpoint_dir(pl_module)
            latest = self._find_latest_epoch_ckpt(checkpoint_dir)
            if latest is not None:
                self._update_symlink(checkpoint_dir / "last.ckpt", latest)
                self.last_checkpoint_path = str(checkpoint_dir / "last.ckpt")
                # Mirror last.json/mp4 if available
                try:
                    epoch_str = latest.stem.split("=")[-1]
                    jpath = checkpoint_dir / f"epoch={epoch_str}.json"
                    if jpath.exists():
                        self._update_symlink(checkpoint_dir / "last.json", jpath)
                    vid = checkpoint_dir / f"epoch={epoch_str}.mp4"
                    if vid.exists():
                        self._update_symlink(checkpoint_dir / "last.mp4", vid)
                except Exception:
                    pass
    
    def on_fit_end(self, trainer, pl_module):
        """Handle training completion summary and ensure symlinks are consistent."""
        # Avoid redundant end-of-training prints; report is handled elsewhere

        # Ensure best/last symlinks are pointing to the latest epoch artifacts
        try:
            ckpt_dir = pl_module.run_manager.ensure_path("checkpoints/")
            latest = self._find_latest_epoch_ckpt(ckpt_dir)
            if latest is not None:
                self._update_symlink(ckpt_dir / "last.ckpt", latest)
                # Update last.json/mp4 if available
                try:
                    epoch_str = latest.stem.split("=")[-1]
                    jpath = ckpt_dir / f"epoch={epoch_str}.json"
                    if jpath.exists():
                        self._update_symlink(ckpt_dir / "last.json", jpath)
                    vid = ckpt_dir / f"epoch={epoch_str}.mp4"
                    if vid.exists():
                        self._update_symlink(ckpt_dir / "last.mp4", vid)
                except Exception:
                    pass

            # If we tracked a best epoch, ensure best symlinks exist
            if self.best_epoch_index is not None:
                best_epoch = int(self.best_epoch_index)
                best_epoch_ckpt = ckpt_dir / f"epoch={best_epoch:02d}.ckpt"
                if best_epoch_ckpt.exists():
                    self._update_symlink(ckpt_dir / "best.ckpt", best_epoch_ckpt)
                    best_json = ckpt_dir / f"epoch={best_epoch:02d}.json"
                    if best_json.exists():
                        self._update_symlink(ckpt_dir / "best.json", best_json)
                    best_vid = ckpt_dir / f"epoch={best_epoch:02d}.mp4"
                    if best_vid.exists():
                        self._update_symlink(ckpt_dir / "best.mp4", best_vid)
        except Exception as e:
            print(f"Warning: final symlink reconciliation failed: {e}")
