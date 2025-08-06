"""Run directory management utilities for organizing all run assets."""

import json
from pathlib import Path
from typing import Optional
from dataclasses import asdict
import wandb


class RunManager:
    """Manages run-specific directories and assets organization."""
    
    def __init__(self, base_runs_dir: str = "runs"):
        """
        Initialize run manager.
        
        Args:
            base_runs_dir: Base directory where all runs will be stored
        """
        self.base_runs_dir = Path(base_runs_dir)
        self.run_dir: Optional[Path] = None
        self.run_id: Optional[str] = None
        
    def setup_run_directory(self, wandb_run: Optional[object] = None) -> Path:
        """
        Setup a run directory using wandb run ID.
        
        Args:
            wandb_run: wandb run object (uses wandb.run if None)
            
        Returns:
            Path to the created run directory
        """
        if wandb_run is None:
            wandb_run = wandb.run
            
        if wandb_run is None:
            raise ValueError("No wandb run available. Make sure wandb.init() has been called.")
            
        self.run_id = wandb_run.id
        self.run_dir = self.base_runs_dir / self.run_id
        
        # Create the run directory
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different asset types
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "videos").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "configs").mkdir(exist_ok=True)
        
        # Create/update latest-run symlink to point to this run
        self._update_latest_run_symlink()
        
        return self.run_dir
    
    def _update_latest_run_symlink(self):
        """Create or update the latest-run symlink to point to the current run directory."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
            
        latest_run_link = self.base_runs_dir / "latest-run"
        
        # Remove existing symlink if it exists
        if latest_run_link.exists() or latest_run_link.is_symlink():
            latest_run_link.unlink()
        
        # Create new symlink pointing to the current run directory
        # Use relative path for the symlink target to make it more portable
        relative_target = Path(self.run_id)
        latest_run_link.symlink_to(relative_target)
    
    def get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory for this run."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir / "checkpoints"
    
    def get_video_dir(self) -> Path:
        """Get the video directory for this run."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir / "videos"
    
    def get_logs_dir(self) -> Path:
        """Get the logs directory for this run."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir / "logs"
    
    def get_configs_dir(self) -> Path:
        """Get the configs directory for this run."""
        if self.run_dir is None:
            raise ValueError("Run directory not set up. Call setup_run_directory() first.")
        return self.run_dir / "configs"
    
    def save_config(self, config, filename: str = "config.json"):
        """
        Save configuration to the run directory.
        
        Args:
            config: Configuration object (should be convertible to dict)
            filename: Name of the config file
        """
        configs_dir = self.get_configs_dir()
        config_path = configs_dir / filename
        
        # Convert config to dict if it's a dataclass
        if hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        else:
            config_dict = config
            
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
            
        return config_path
    
    def get_run_info(self) -> dict:
        """Get information about the current run."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "base_runs_dir": str(self.base_runs_dir)
        }
