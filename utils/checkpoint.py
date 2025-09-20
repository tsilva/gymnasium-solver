"""Checkpoint management utilities for model saving and resuming training.

Set environment variable `VIBES_QUIET=1` (or `VIBES_DISABLE_CHECKPOINT_LOGS=1`)
to suppress informational prints emitted during checkpoint loading.
"""

import os
from pathlib import Path
from typing import Any, Dict

import torch


def _quiet_mode_enabled() -> bool:
    flag1 = os.environ.get("VIBES_QUIET", "").strip().lower()
    flag2 = os.environ.get("VIBES_DISABLE_CHECKPOINT_LOGS", "").strip().lower()
    return flag1 in {"1", "true", "yes", "on"} or flag2 in {"1", "true", "yes", "on"}


def load_checkpoint(checkpoint_path: Path, agent, resume_training: bool = True) -> Dict[str, Any]:
    """Load a checkpoint and optionally restore optimizer/RNG/state."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not _quiet_mode_enabled():
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
    
    if not _quiet_mode_enabled():
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
