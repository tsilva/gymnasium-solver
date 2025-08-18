"""Checkpoint management utilities for model saving and resuming training."""

from pathlib import Path

import pytorch_lightning as pl
import torch


class ModelCheckpointCallback(pl.Callback):
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
        from dataclasses import asdict
        
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
        checkpoint_data = {
            'model_state_dict': agent.policy_model.state_dict(),
            'optimizer_state_dict': agent.optimizers().state_dict() if hasattr(agent.optimizers(), 'state_dict') else None,
            'config_dict': asdict(agent.config),
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
        return checkpoint_path
    
    def _should_save_best(self, current_value: float) -> bool:
        """Check if current metric value is better than the best seen so far."""
        if self.mode == 'max':
            return current_value > self.best_metric_value
        else:
            return current_value < self.best_metric_value
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Save checkpoints when validation ends (after eval) and handle early stopping."""

        logged_metrics = trainer.logged_metrics
        assert self.monitor in logged_metrics, f"Monitor metric '{self.monitor}' not found in logged metrics: {trainer.logged_metrics.keys()}"
        
        current_metric_value = None
        if self.monitor in logged_metrics:
            current_metric_value = float(logged_metrics[self.monitor])

        if current_metric_value is None:
            # Nothing to monitor this epoch
            self._handle_early_stopping_and_tracking(trainer, pl_module)
            return

        checkpoint_dir = self._get_checkpoint_dir(pl_module)

        # Best model logic
        saved_timestamped_path = None  # track if we saved an epoch checkpoint in this call
        is_best = self._should_save_best(current_metric_value)
        if is_best:
            self.best_metric_value = current_metric_value
            epoch = pl_module.current_epoch
            step = pl_module.global_step
            timestamped_path = checkpoint_dir / f"epoch={epoch:02d}.ckpt"
            self._save_checkpoint(
                pl_module,
                timestamped_path,
                is_best=True,
                metrics=logged_metrics,
                current_eval_reward=current_metric_value,
            )
            saved_timestamped_path = timestamped_path

            best_path = checkpoint_dir / "best.ckpt"
            self.best_checkpoint_path = self._save_checkpoint(
                pl_module,
                best_path,
                is_best=True,
                metrics=logged_metrics,
                current_eval_reward=current_metric_value,
            )

            print(f"New best model saved with {self.monitor}={current_metric_value:.4f}")
            print(f"  Timestamped: {timestamped_path}")
            print(f"  Best: {best_path}")
            pl_module.best_model_path = str(best_path)
            pl_module.best_eval_reward = current_metric_value

        # Unified reward threshold (config overrides env spec, then gym.spec fallback)
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

        if (self.save_threshold_reached and
            effective_threshold is not None and
            current_metric_value >= effective_threshold and
            not self._threshold_checkpoint_saved):
            epoch = pl_module.current_epoch
            step = pl_module.global_step
            if saved_timestamped_path is not None:
                # Avoid redundancy: annotate the just-saved epoch checkpoint with threshold info
                self._save_checkpoint(
                    pl_module,
                    saved_timestamped_path,
                    is_best=True,
                    is_threshold=True,
                    metrics=logged_metrics,
                    current_eval_reward=current_metric_value,
                    threshold_value=effective_threshold,
                )
                print(
                    f"Threshold reached! Annotated checkpoint {saved_timestamped_path.name} "
                    f"with {self.monitor}={current_metric_value:.4f} (threshold={effective_threshold})"
                )
            else:
                # No epoch checkpoint saved this validation; create a dedicated threshold checkpoint
                threshold_path = checkpoint_dir / f"threshold-epoch={epoch:02d}-step={step:04d}.ckpt"
                self._save_checkpoint(
                    pl_module,
                    threshold_path,
                    is_threshold=True,
                    metrics=logged_metrics,
                    current_eval_reward=current_metric_value,
                    threshold_value=effective_threshold,
                )
                print(
                    f"Threshold reached! Saved model with {self.monitor}={current_metric_value:.4f} "
                    f"(threshold={effective_threshold}) at {threshold_path}"
                )
            self._threshold_checkpoint_saved = True
            if not trainer.should_stop and getattr(pl_module.config, 'early_stop_on_eval_threshold', True):
                print(f"Early stopping at epoch {pl_module.current_epoch} with eval mean reward {current_metric_value:.2f} >= threshold {effective_threshold}")
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
            if reward_threshold is not None:
                print(f"Using environment spec reward_threshold: {reward_threshold}")
            else:
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
        """Save last checkpoint after each training epoch."""
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
            last_path = checkpoint_dir / "last.ckpt"
            # Capture whatever metrics are available at train epoch end
            logged_metrics = getattr(trainer, 'logged_metrics', None)
            cur_eval = None
            try:
                if isinstance(logged_metrics, dict) and self.monitor in logged_metrics:
                    cur_eval = float(logged_metrics[self.monitor])
            except Exception:
                cur_eval = None
            self.last_checkpoint_path = self._save_checkpoint(
                pl_module,
                last_path,
                is_last=True,
                metrics=logged_metrics,
                current_eval_reward=cur_eval,
            )
    
    def on_fit_end(self, trainer, pl_module):
        """Handle training completion summary that was previously in BaseAgent."""
        # Print completion summary
        if self.best_checkpoint_path:
            print(f"Best model saved at {self.best_checkpoint_path} with eval reward {pl_module.best_eval_reward:.2f}")
        elif self.last_checkpoint_path:
            print(f"Last model saved at {self.last_checkpoint_path}")
        else:
            print("No checkpoints were saved during training")

    # After training finishes, run a final evaluation using the best checkpoint
    # and record a full-length video saved as 'best.mp4'.
        try:
            best_ckpt = self.best_checkpoint_path
            if best_ckpt is None:
                # If no explicit best was tracked, try conventional path in checkpoint_dir
                candidate = Path(self.checkpoint_dir) / "best.ckpt"
                if candidate.exists():
                    best_ckpt = candidate

            if best_ckpt is None or not Path(best_ckpt).exists():
                # Nothing to do if we don't have a best checkpoint
                return

            # Load the best checkpoint weights (no need to resume optimizer/state)
            try:
                from utils.checkpoint import load_checkpoint
                load_checkpoint(Path(best_ckpt), pl_module, resume_training=False)
            except Exception as e:
                print(f"Warning: failed to load best checkpoint for final evaluation: {e}")
                return

            # Prepare video path under the run's video directory
            # Store alongside other eval videos: runs/<run_id>/videos/eval/best.mp4
            video_dir = pl_module.run_manager.get_video_dir() / "eval"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / "best.mp4"

            # Ensure full-length recording by temporarily disabling video_length cap
            # validation_env is a VecInfoWrapper around VecVideoRecorder -> access via .venv
            vec_rec = getattr(pl_module.validation_env, 'venv', None)
            old_len = None
            if vec_rec is not None and hasattr(vec_rec, 'video_length'):
                old_len = getattr(vec_rec, 'video_length')
                try:
                    setattr(vec_rec, 'video_length', None)
                except Exception:
                    pass

            # Run a one-episode deterministic evaluation while recording
            try:
                from utils.evaluation import evaluate_policy
                with pl_module.validation_env.recorder(str(video_path), record_video=True):
                    _ = evaluate_policy(
                        pl_module.validation_env,
                        pl_module.policy_model,
                        n_episodes=1,
                        deterministic=getattr(pl_module.config, 'eval_deterministic', True),
                    )
                print(f"Saved final evaluation video to: {video_path}")
            except Exception as e:
                print(f"Warning: failed to record final evaluation video: {e}")
            finally:
                # Restore original video length cap if we changed it
                if vec_rec is not None and hasattr(vec_rec, 'video_length'):
                    try:
                        setattr(vec_rec, 'video_length', old_len)
                    except Exception:
                        pass
        except Exception as e:
            # Never fail training teardown due to best-checkpoint evaluation
            print(f"Warning: final best-checkpoint evaluation skipped due to error: {e}")
