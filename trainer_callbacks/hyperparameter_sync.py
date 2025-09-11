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
import queue
import threading
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from utils.io import read_json

class HyperparamSyncCallback(pl.Callback):
    """
    Callback that enables manual hyperparameter adjustment during training.

    Features:
    - Real-time hyperparameter adjustment via file monitoring
    - Manual triggers through a control JSON file
    """
    
    def __init__(
        self,
        control_dir: str | None = None,
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

        run_dir = pl_module.run_manager.get_run_dir()
        assert run_dir.exists(), f"Run directory {run_dir} does not exist."

        self.control_dir = Path(run_dir)
        self.control_file = self.control_dir / "config.json"
        assert self.control_file.exists(), f"Control file {self.control_file} does not exist."

        # Store original hyperparameters
        self.original_hyperparams = {
            'policy_lr': pl_module.config.policy_lr,
            'ent_coef': pl_module.config.ent_coef,
            'clip_range': getattr(pl_module.config, 'clip_range', None),
            'vf_coef': getattr(pl_module.config, 'vf_coef', None),
            'max_grad_norm': pl_module.config.max_grad_norm,
        }

        # Start background monitoring if enabled
        if self.enable_manual_control:
            self.start_monitoring()

        # Log initial hyperparameters to W&B under train namespace
        self._log_hyperparams(pl_module)

        if self.verbose:
            print(f"\n🎛️  Hyperparameter manual control enabled!")
            print(f"   Control directory: {self.control_dir}")
            print(f"   Control file: {self.control_file.name}")
            print(f"   Edit this file to adjust hyperparameters during training.")
    
    
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
            # Check for file modification
            if self.control_file.exists():
                current_mtime = self.control_file.stat().st_mtime
                
                if current_mtime > self.last_check_time:
                    self.last_check_time = current_mtime
                    try:
                        data = read_json(self.control_file)
                        self.adjustment_queue.put(data)
                    except (json.JSONDecodeError, IOError) as e:
                        if self.verbose:
                            print(f"⚠️  Error reading {self.control_file.name}: {e}")
            
            # Sleep until next check
            self.stop_monitoring.wait(self.check_interval)
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Process any pending hyperparameter adjustments."""
        # Process adjustments from file monitoring
        while not self.adjustment_queue.empty():
            try:
                data = self.adjustment_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_adjustments(trainer, pl_module, data)
    
    def _apply_adjustments(self, trainer: pl.Trainer, pl_module: pl.LightningModule, data: Dict[str, Any]) -> None:
        """Apply hyperparameter adjustments from control file."""
        changes = []
        changed_for_log: Dict[str, float] = {}
        
        # Learning rate (support either 'policy_lr' or 'policy_lr' key from config.json)
        if "policy_lr" in data or "policy_lr" in data:
            lr_key = "policy_lr" if "policy_lr" in data else "policy_lr"
            new_lr = float(data[lr_key])
            old_lr = pl_module.config.policy_lr
            if abs(new_lr - old_lr) > 1e-8:
                self._update_policy_lr(trainer, pl_module, new_lr)
                changes.append(f"policy_lr: {old_lr:.2e} → {new_lr:.2e}")
                changed_for_log["policy_lr"] = new_lr
        
        # Entropy coefficient
        if "ent_coef" in data:
            new_val = float(data["ent_coef"])
            old_val = pl_module.config.ent_coef
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.ent_coef = new_val
                changes.append(f"ent_coef: {old_val:.3f} → {new_val:.3f}")
                changed_for_log["ent_coef"] = new_val
        
        # Gradient clipping
        if "max_grad_norm" in data:
            new_val = float(data["max_grad_norm"])
            old_val = pl_module.config.max_grad_norm
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.max_grad_norm = new_val
                changes.append(f"max_grad_norm: {old_val:.3f} → {new_val:.3f}")
                changed_for_log["max_grad_norm"] = new_val
        
        # PPO-specific parameters
        if hasattr(pl_module.config, 'clip_range') and "clip_range" in data:
            new_val = float(data["clip_range"])
            old_val = pl_module.config.clip_range
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.clip_range = new_val
                changes.append(f"clip_range: {old_val:.3f} → {new_val:.3f}")
                changed_for_log["clip_range"] = new_val
        
        if hasattr(pl_module.config, 'vf_coef') and "vf_coef" in data:
            new_val = float(data["vf_coef"])
            old_val = pl_module.config.vf_coef
            if abs(new_val - old_val) > 1e-8:
                pl_module.config.vf_coef = new_val
                changes.append(f"vf_coef: {old_val:.3f} → {new_val:.3f}")
                changed_for_log["vf_coef"] = new_val

        # Emit W&B logs for changed hyperparameters under train namespace
        if changed_for_log:
            pl_module.metrics.record("train", changed_for_log)

        if changes and self.verbose:
            print(f"🎛️  Hyperparameters updated (epoch {trainer.current_epoch}): {', '.join(changes)}")
    
    def _update_policy_lr(self, trainer: pl.Trainer, pl_module: pl.LightningModule, new_lr: float) -> None:
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

    def _log_hyperparams(self, pl_module: pl.LightningModule) -> None:
        """Helper to log current hyperparameters under train namespace."""
        hp: Dict[str, float] = {}
        hp["policy_lr"] = float(pl_module.config.policy_lr)
        for key in ("ent_coef", "vf_coef", "clip_range", "max_grad_norm"):
            if hasattr(pl_module.config, key) and getattr(pl_module.config, key) is not None:
                hp[key] = float(getattr(pl_module.config, key))
        if hp:
            pl_module.metrics.record_train(hp)
    
    
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Clean up monitoring when training ends."""
        self.stop_monitoring_thread()

    def reset_to_original(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset all hyperparameters to their original values."""
        if self.verbose:
            print("🔄 Resetting hyperparameters to original values...")
        
        # Reset learning rate
        if 'policy_lr' in self.original_hyperparams:
            self._update_policy_lr(trainer, pl_module, self.original_hyperparams['policy_lr'])
        
        # Reset other hyperparameters
        for key, value in self.original_hyperparams.items():
            if value is not None and hasattr(pl_module.config, key.replace('policy_lr', 'policy_lr')):
                setattr(pl_module.config, key.replace('policy_lr', 'policy_lr'), value)
