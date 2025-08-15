"""
Manual hyperparameter control callback for RL training.

This callback enables on-the-fly adjustment of selected hyperparameters via a
control JSON file. Any change to the control file is detected and applied at
the start of the next training epoch.

Notes:
- Automatic/metric-based scheduling has been removed. This utility is solely
    for manual control during experiments.
"""

import json
import time
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from pathlib import Path
import threading
import queue


class HyperparameterScheduler(pl.Callback):
    """
    Callback that enables manual hyperparameter adjustment during training.

    Features:
    - Real-time hyperparameter adjustment via file monitoring
    - Manual triggers through a control JSON file
    """
    
    def __init__(
        self,
        control_dir: Optional[str] = None,
        check_interval: float = 5.0,
        enable_lr_scheduling: bool = False,
        enable_manual_control: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the manual hyperparameter controller.
        
        Args:
            control_dir: Directory to monitor for control files. If None, uses run directory.
            check_interval: How often to check for control files (seconds)
            enable_lr_scheduling: Deprecated; retained for backward-compatibility (ignored)
            enable_manual_control: Enable manual control via files
            verbose: Print adjustment messages
        """
        super().__init__()
        self.control_dir = control_dir
        self.check_interval = check_interval
        # Deprecated: kept for backward-compatibility; no automatic scheduling is performed
        self.enable_lr_scheduling = enable_lr_scheduling
        self.enable_manual_control = enable_manual_control
        self.verbose = verbose
        
        # Control file path (set in on_fit_start)
        self.control_file = None
        
        # Last modification time for file monitoring
        self.last_check_time = 0
        
        # Original hyperparameters for reference
        self.original_hyperparams = {}
        
        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.adjustment_queue = queue.Queue()
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize control files and start monitoring."""
        # Set up control directory
        if self.control_dir is None:
            if hasattr(pl_module, 'run_manager') and pl_module.run_manager:
                self.control_dir = pl_module.run_manager.run_dir / "hyperparam_control"
            else:
                self.control_dir = Path("./runs/latest-run/hyperparam_control")
        else:
            self.control_dir = Path(self.control_dir)
        
        self.control_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up control file path
        self.control_file = self.control_dir / "hyperparameters.json"
        
        # Store original hyperparameters
        self.original_hyperparams = {
            'learning_rate': pl_module.config.policy_lr,
            'ent_coef': pl_module.config.ent_coef,
            'clip_range': getattr(pl_module.config, 'clip_range', None),
            'vf_coef': getattr(pl_module.config, 'vf_coef', None),
            'max_grad_norm': pl_module.config.max_grad_norm,
        }
        
        # Create example control file
        self._create_control_file(pl_module)
        
        # Start background monitoring if enabled
        if self.enable_manual_control:
            self.start_monitoring()
        
        if self.verbose:
            print(f"\n🎛️  Hyperparameter manual control enabled!")
            print(f"   Control directory: {self.control_dir}")
            print(f"   Control file: {self.control_file.name}")
            print(f"   Edit this file to adjust hyperparameters during training.")
    
    def _create_control_file(self, pl_module: pl.LightningModule) -> None:
        """Create the unified control file with current values."""
        control_data = {
            # Core hyperparameters
            "learning_rate": pl_module.config.policy_lr,
            "ent_coef": pl_module.config.ent_coef,
            "max_grad_norm": pl_module.config.max_grad_norm,
            
            # Algorithm-specific parameters (PPO)
            "clip_range": getattr(pl_module.config, 'clip_range', None),
            "vf_coef": getattr(pl_module.config, 'vf_coef', None),
            
            # Metadata
            "description": "Modify any hyperparameter in this file. Changes are applied at the start of the next epoch.",
            "supported_params": [
                "learning_rate - Learning rate for optimizer",
                "ent_coef - Entropy coefficient", 
                "max_grad_norm - Maximum gradient norm for clipping",
                "clip_range - PPO clipping range (PPO only)",
                "vf_coef - Value function coefficient (PPO only)"
            ],
            "last_modified": time.time()
        }
        
        # Remove None values for cleaner file
        if control_data["clip_range"] is None:
            del control_data["clip_range"]
        if control_data["vf_coef"] is None:
            del control_data["vf_coef"]
        
        # Write file only if it doesn't exist
        if not self.control_file.exists():
            with open(self.control_file, 'w') as f:
                json.dump(control_data, f, indent=2)
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_thread is None:
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_files, daemon=True)
            self.monitoring_thread.start()
    
    def stop_monitoring_thread(self) -> None:
        """Stop background monitoring thread."""
        if self.monitoring_thread is not None:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=1.0)
            self.monitoring_thread = None
    
    def _monitor_files(self) -> None:
        """Background thread to monitor control file for changes."""
        while not self.stop_monitoring.is_set():
            try:
                # Check for file modification
                if self.control_file.exists():
                    current_mtime = self.control_file.stat().st_mtime
                    
                    if current_mtime > self.last_check_time:
                        self.last_check_time = current_mtime
                        try:
                            with open(self.control_file, 'r') as f:
                                data = json.load(f)
                            self.adjustment_queue.put(data)
                        except (json.JSONDecodeError, IOError) as e:
                            if self.verbose:
                                print(f"⚠️  Error reading {self.control_file.name}: {e}")
                
                # Sleep until next check
                self.stop_monitoring.wait(self.check_interval)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error in file monitoring: {e}")
                self.stop_monitoring.wait(self.check_interval)
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Process any pending hyperparameter adjustments."""
        # Process adjustments from file monitoring
        while not self.adjustment_queue.empty():
            try:
                data = self.adjustment_queue.get_nowait()
                self._apply_adjustments(trainer, pl_module, data)
            except queue.Empty:
                break
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error applying adjustment: {e}")
    
    def _apply_adjustments(self, trainer: pl.Trainer, pl_module: pl.LightningModule, data: Dict[str, Any]) -> None:
        """Apply hyperparameter adjustments from control file."""
        changes = []
        
        # Learning rate
        if "learning_rate" in data:
            new_lr = float(data["learning_rate"])
            old_lr = pl_module.config.policy_lr
            if abs(new_lr - old_lr) > 1e-8:
                self._update_learning_rate(trainer, pl_module, new_lr)
                changes.append(f"learning_rate: {old_lr:.2e} → {new_lr:.2e}")
        
        # Entropy coefficient
        if "ent_coef" in data:
            new_val = float(data["ent_coef"])
            old_val = pl_module.config.ent_coef
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.ent_coef = new_val
                changes.append(f"ent_coef: {old_val:.3f} → {new_val:.3f}")
        
        # Gradient clipping
        if "max_grad_norm" in data:
            new_val = float(data["max_grad_norm"])
            old_val = pl_module.config.max_grad_norm
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.max_grad_norm = new_val
                changes.append(f"max_grad_norm: {old_val:.3f} → {new_val:.3f}")
        
        # PPO-specific parameters
        if hasattr(pl_module.config, 'clip_range') and "clip_range" in data:
            new_val = float(data["clip_range"])
            old_val = pl_module.config.clip_range
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.clip_range = new_val
                changes.append(f"clip_range: {old_val:.3f} → {new_val:.3f}")
        
        if hasattr(pl_module.config, 'vf_coef') and "vf_coef" in data:
            new_val = float(data["vf_coef"])
            old_val = pl_module.config.vf_coef
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.vf_coef = new_val
                changes.append(f"vf_coef: {old_val:.3f} → {new_val:.3f}")
        
        if changes and self.verbose:
            print(f"🎛️  Hyperparameters updated (epoch {trainer.current_epoch}): {', '.join(changes)}")
    
    def _update_learning_rate(self, trainer: pl.Trainer, pl_module: pl.LightningModule, new_lr: float) -> None:
        """Update the learning rate of all optimizers."""
        optimizers = trainer.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        
        old_lr = pl_module.config.policy_lr
        
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # Update config
        pl_module.config.policy_lr = new_lr
        
        if self.verbose and abs(new_lr - old_lr) > 1e-8:
            print(f"📈 Learning rate updated: {old_lr:.2e} → {new_lr:.2e} (epoch {trainer.current_epoch})")
    
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up monitoring when training ends."""
        self.stop_monitoring_thread()
        
        # TODO: this is innacurate
        if self.verbose:
            print(f"\n📊 Final hyperparameters:")
            print(f"   Learning rate: {pl_module.config.policy_lr:.2e}")
            print(f"   Entropy coef: {pl_module.config.ent_coef:.3f}")
            print(f"   Max grad norm: {pl_module.config.max_grad_norm:.3f}")
            if hasattr(pl_module.config, 'clip_range'):
                print(f"   Clip range: {pl_module.config.clip_range:.3f}")
            if hasattr(pl_module.config, 'vf_coef'):
                print(f"   Value function coef: {pl_module.config.vf_coef:.3f}")
    
    def reset_to_original(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset all hyperparameters to their original values."""
        if self.verbose:
            print("🔄 Resetting hyperparameters to original values...")
        
        # Reset learning rate
        if 'learning_rate' in self.original_hyperparams:
            self._update_learning_rate(trainer, pl_module, self.original_hyperparams['learning_rate'])
        
        # Reset other hyperparameters
        for key, value in self.original_hyperparams.items():
            if value is not None and hasattr(pl_module.config, key.replace('learning_rate', 'policy_lr')):
                setattr(pl_module.config, key.replace('learning_rate', 'policy_lr'), value)


# TODO: remove scheduling concerns this is just to support manual control