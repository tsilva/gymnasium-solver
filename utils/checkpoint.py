"""Checkpoint management utilities for model saving and resuming training."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def find_latest_checkpoint(algo_id: str, env_id: str, checkpoint_dir: str = "checkpoints") -> Optional[Path]:
    """Find the latest checkpoint for a given algorithm and environment.

    Supports both legacy layout checkpoints/<algo>/<env>/... and run-local
    checkpoints saved under runs/<run_id>/checkpoints/ when consuming directly.
    """
    env_id_clean = env_id.replace('/', '_').replace('\\', '_')
    checkpoint_path = Path(checkpoint_dir) / algo_id / env_id_clean
    
    if not checkpoint_path.exists():
        return None
    
    # Look for checkpoints in order of preference: best -> threshold -> last -> latest timestamped
    for checkpoint_name in ["best.ckpt", "best_checkpoint.ckpt", "last.ckpt", "last_checkpoint.ckpt"]:
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
    # Support both legacy epoch=xx-step=yy.ckpt and new epoch=xx.ckpt
    timestamped_checkpoints = list(checkpoint_path.glob("epoch=*-step=*.ckpt")) or list(checkpoint_path.glob("epoch=*.ckpt"))
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
                    0 if x == "best.ckpt" else
                    1 if x == "last.ckpt" else
                    2 if x.startswith("threshold-") else
                    3
                ))
                checkpoints[algo_id][env_id] = env_checkpoints
    
    return checkpoints
