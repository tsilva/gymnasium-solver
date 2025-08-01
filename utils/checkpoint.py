"""Checkpoint management utilities for model saving and resuming training."""

import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict


class ModelCheckpointCallback(pl.Callback):
    """Custom checkpoint callback that saves models when eval reward improves."""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 monitor: str = "eval/ep_rew_mean",
                 mode: str = "max",
                 save_last: bool = True,
                 save_threshold_reached: bool = True):
        """
        Initialize the checkpoint callback.
        
        Args:
            checkpoint_dir: Base directory for checkpoints
            monitor: Metric to monitor for saving best checkpoints
            mode: 'max' or 'min' for the monitored metric
            save_last: Whether to save the last checkpoint
            save_threshold_reached: Whether to save when reward threshold is reached
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.save_threshold_reached = save_threshold_reached
        
        self.best_metric_value = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None
        
    def _get_checkpoint_dir(self, agent) -> Path:
        """Get the checkpoint directory for this specific agent/env combination."""
        algo_id = agent.config.algo_id
        env_id = agent.config.env_id.replace('/', '_').replace('\\', '_')
        checkpoint_path = self.checkpoint_dir / algo_id / env_id
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
        """Save checkpoints when validation ends (after eval)."""
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
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Save last checkpoint after each training epoch."""
        if self.save_last:
            checkpoint_dir = self._get_checkpoint_dir(pl_module)
            last_path = checkpoint_dir / "last_checkpoint.ckpt"
            self.last_checkpoint_path = self._save_checkpoint(
                pl_module, last_path, is_last=True
            )


def find_latest_checkpoint(algo_id: str, env_id: str, checkpoint_dir: str = "checkpoints") -> Optional[Path]:
    """Find the latest checkpoint for a given algorithm and environment."""
    env_id_clean = env_id.replace('/', '_').replace('\\', '_')
    checkpoint_path = Path(checkpoint_dir) / algo_id / env_id_clean
    
    if not checkpoint_path.exists():
        return None
    
    # Look for checkpoints in order of preference: best -> threshold -> last -> latest timestamped
    for checkpoint_name in ["best_checkpoint.ckpt", "last_checkpoint.ckpt"]:
        checkpoint_file = checkpoint_path / checkpoint_name
        if checkpoint_file.exists():
            return checkpoint_file
    
    # Look for threshold checkpoints (timestamped)
    threshold_checkpoints = list(checkpoint_path.glob("threshold-epoch=*-step=*.ckpt"))
    if threshold_checkpoints:
        # Sort by modification time and return the latest
        threshold_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return threshold_checkpoints[0]
    
    # Look for any timestamped checkpoints (epoch=XX-step=XXXX.ckpt)
    timestamped_checkpoints = list(checkpoint_path.glob("epoch=*-step=*.ckpt"))
    if timestamped_checkpoints:
        # Sort by modification time and return the latest
        timestamped_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return timestamped_checkpoints[0]
    
    return None


def load_checkpoint(checkpoint_path: Path, agent, resume_training: bool = True) -> Dict[str, Any]:
    """
    Load a checkpoint and optionally resume training state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        agent: The agent to load the checkpoint into
        resume_training: Whether to resume training state (optimizer, epoch, etc.)
        
    Returns:
        Dictionary with checkpoint metadata
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load model state
    agent.policy_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load training state if resuming
    if resume_training:
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer = agent.optimizers()
            if hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore RNG states for reproducibility
        if 'rng_states' in checkpoint:
            torch.set_rng_state(checkpoint['rng_states']['torch'])
            if checkpoint['rng_states']['torch_cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['rng_states']['torch_cuda'])
        
        # Restore training counters
        if hasattr(agent, 'total_timesteps'):
            agent.total_timesteps = checkpoint.get('total_timesteps', 0)
        if hasattr(agent, 'best_eval_reward'):
            agent.best_eval_reward = checkpoint.get('best_eval_reward', float('-inf'))
    
    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    total_timesteps = checkpoint.get('total_timesteps', 'unknown')
    best_reward = checkpoint.get('best_eval_reward', 'unknown')
    current_reward = checkpoint.get('current_eval_reward', 'unknown')
    
    print(f"Checkpoint loaded:")
    print(f"  Epoch: {epoch}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Best eval reward: {best_reward}")
    print(f"  Current eval reward: {current_reward}")
    print(f"  Is best: {checkpoint.get('is_best', False)}")
    print(f"  Is threshold: {checkpoint.get('is_threshold', False)}")
    
    return checkpoint


def list_available_checkpoints(checkpoint_dir: str = "checkpoints") -> Dict[str, Dict[str, list]]:
    """List all available checkpoints organized by algorithm and environment."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return {}
    
    checkpoints = {}
    for algo_dir in checkpoint_path.iterdir():
        if not algo_dir.is_dir():
            continue
            
        algo_id = algo_dir.name
        checkpoints[algo_id] = {}
        
        for env_dir in algo_dir.iterdir():
            if not env_dir.is_dir():
                continue
                
            env_id = env_dir.name
            env_checkpoints = []
            
            # Collect all checkpoint files
            for checkpoint_file in env_dir.glob("*.ckpt"):
                env_checkpoints.append(checkpoint_file.name)
            
            if env_checkpoints:
                # Sort checkpoints: best first, then timestamped by name
                env_checkpoints.sort(key=lambda x: (
                    0 if x == "best_checkpoint.ckpt" else
                    1 if x == "last_checkpoint.ckpt" else
                    2 if x.startswith("threshold-") else
                    3
                ))
                checkpoints[algo_id][env_id] = env_checkpoints
    
    return checkpoints
