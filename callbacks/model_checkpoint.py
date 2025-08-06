"""Checkpoint management utilities for model saving and resuming training."""

import torch
import pytorch_lightning as pl
from pathlib import Path


class ModelCheckpointCallback(pl.Callback):
    """Custom checkpoint callback that handles all model checkpointing logic including resume."""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 monitor: str = "eval/ep_rew_mean",
                 mode: str = "max",
                 save_last: bool = True,
                 save_threshold_reached: bool = True,
                 resume: bool = False):
        """
        Initialize the checkpoint callback.
        
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
    
    def _save_checkpoint(self, agent, checkpoint_path: Path, is_best: bool = False, is_last: bool = False, is_threshold: bool = False):
        """Save a checkpoint with all necessary information."""
        from dataclasses import asdict
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': agent.policy_model.state_dict(),
            'optimizer_state_dict': agent.optimizers().state_dict() if hasattr(agent.optimizers(), 'state_dict') else None,
            'config_dict': asdict(agent.config),
            'epoch': agent.current_epoch,
            'global_step': agent.global_step,
            'total_timesteps': getattr(agent, 'total_timesteps', 0),
            'best_eval_reward': getattr(agent, 'best_eval_reward', float('-inf')),
            'current_eval_reward': None,  # Will be set by caller if available
            'is_best': is_best,
            'is_last': is_last,
            'is_threshold': is_threshold,
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
        if not hasattr(trainer, 'logged_metrics'):
            return
            
        logged_metrics = trainer.logged_metrics
        
        # Get the monitored metric value
        current_metric_value = None
        if self.monitor in logged_metrics:
            current_metric_value = float(logged_metrics[self.monitor])
        
        if current_metric_value is not None:
            checkpoint_dir = self._get_checkpoint_dir(pl_module)
            
            # Check if this is the best model so far
            is_best = self._should_save_best(current_metric_value)
            
            if is_best:
                self.best_metric_value = current_metric_value
                
                # Save timestamped checkpoint with epoch-step format
                epoch = pl_module.current_epoch
                step = pl_module.global_step
                timestamped_path = checkpoint_dir / f"epoch={epoch:02d}-step={step:04d}.ckpt"
                self._save_checkpoint(
                    pl_module, timestamped_path, is_best=True
                )
                checkpoint_data = torch.load(timestamped_path)
                checkpoint_data['current_eval_reward'] = current_metric_value
                torch.save(checkpoint_data, timestamped_path)
                
                # Also save/overwrite the best checkpoint
                best_path = checkpoint_dir / "best_checkpoint.ckpt"
                self.best_checkpoint_path = self._save_checkpoint(
                    pl_module, best_path, is_best=True
                )
                checkpoint_data = torch.load(best_path)
                checkpoint_data['current_eval_reward'] = current_metric_value
                torch.save(checkpoint_data, best_path)
                
                print(f"New best model saved with {self.monitor}={current_metric_value:.4f}")
                print(f"  Timestamped: {timestamped_path}")
                print(f"  Best: {best_path}")
                
                # Update the agent's best model path for compatibility
                pl_module.best_model_path = str(best_path)
                pl_module.best_eval_reward = current_metric_value
            
            # Check if reward threshold is reached and save threshold checkpoint
            if (self.save_threshold_reached and 
                hasattr(pl_module.config, 'reward_threshold') and 
                pl_module.config.reward_threshold is not None and
                current_metric_value >= pl_module.config.reward_threshold):
                
                epoch = pl_module.current_epoch
                step = pl_module.global_step
                threshold_path = checkpoint_dir / f"threshold-epoch={epoch:02d}-step={step:04d}.ckpt"
                self._save_checkpoint(
                    pl_module, threshold_path, is_threshold=True
                )
                checkpoint_data = torch.load(threshold_path)
                checkpoint_data['current_eval_reward'] = current_metric_value
                torch.save(checkpoint_data, threshold_path)
                
                print(f"Threshold reached! Saved model with reward {current_metric_value:.4f} at {threshold_path}")
                
                # Early stopping when threshold is reached
                print(f"Early stopping at epoch {pl_module.current_epoch} with eval mean reward {current_metric_value:.2f} >= threshold {pl_module.config.reward_threshold}")
                trainer.should_stop = True
        
        # Handle early stopping logic and best reward tracking that was in BaseAgent
        self._handle_early_stopping_and_tracking(trainer, pl_module)
    
    def _handle_early_stopping_and_tracking(self, trainer, pl_module):
        """Handle early stopping logic and best reward tracking that was previously in BaseAgent."""
        # Get reward threshold - use config if provided, otherwise use environment's reward threshold
        config_threshold = pl_module.config.reward_threshold
        if config_threshold is not None: 
            reward_threshold = config_threshold
            print(f"Using config reward_threshold: {reward_threshold}")
        else: 
            reward_threshold = pl_module.validation_env.get_reward_threshold()
            if reward_threshold is not None:
                print(f"Using environment spec reward_threshold: {reward_threshold}")

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
                if ep_rew_mean >= reward_threshold and not trainer.should_stop:
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
        if self.save_last:
            checkpoint_dir = self._get_checkpoint_dir(pl_module)
            last_path = checkpoint_dir / "last_checkpoint.ckpt"
            self.last_checkpoint_path = self._save_checkpoint(
                pl_module, last_path, is_last=True
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
