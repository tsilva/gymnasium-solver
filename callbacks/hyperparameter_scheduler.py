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
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl


class HyperparamSyncCallback(pl.Callback):
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
                self.control_dir = pl_module.run_manager.run_dir
            else:
                self.control_dir = Path("./runs/latest-run")
        else:
            self.control_dir = Path(self.control_dir)

        self.control_dir.mkdir(parents=True, exist_ok=True)

        # Monitor the run's main configuration file for changes
        # Users can edit this file during training to tweak hyperparameters.
        self.control_file = self.control_dir / "config.json"

        # Store original hyperparameters
        self.original_hyperparams = {
            'learning_rate': pl_module.config.policy_lr,
            'ent_coef': pl_module.config.ent_coef,
            'clip_range': getattr(pl_module.config, 'clip_range', None),
            'vf_coef': getattr(pl_module.config, 'vf_coef', None),
            'max_grad_norm': pl_module.config.max_grad_norm,
        }

        # Start background monitoring if enabled
        if self.enable_manual_control:
            self.start_monitoring()

        # Log initial hyperparameters to W&B under train/hyperparams
        try:
            self._log_hyperparams(pl_module)
        except Exception:
            # Never block training on telemetry issues
            pass

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
        changed_for_log: Dict[str, float] = {}
        
        # Learning rate (support either 'policy_lr' or 'learning_rate' key from config.json)
        if "policy_lr" in data or "learning_rate" in data:
            lr_key = "policy_lr" if "policy_lr" in data else "learning_rate"
            new_lr = float(data[lr_key])
            old_lr = pl_module.config.policy_lr
            if abs(new_lr - old_lr) > 1e-8:
                self._update_learning_rate(trainer, pl_module, new_lr)
                changes.append(f"policy_lr: {old_lr:.2e} → {new_lr:.2e}")
                changed_for_log["learning_rate"] = new_lr
        
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

        # Emit W&B logs for changed hyperparameters under train/hyperparams
        if changed_for_log:
            try:
                pl_module.log_metrics(changed_for_log, prefix="train/hyperparams")
            except Exception:
                pass

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

    def _log_hyperparams(self, pl_module: pl.LightningModule) -> None:
        """Helper to log current hyperparameters under train/hyperparams."""
        hp: Dict[str, float] = {}
        try:
            hp["learning_rate"] = float(pl_module.config.policy_lr)
        except Exception:
            pass
        for key in ("ent_coef", "vf_coef", "clip_range", "max_grad_norm"):
            if hasattr(pl_module.config, key) and getattr(pl_module.config, key) is not None:
                try:
                    hp[key] = float(getattr(pl_module.config, key))
                except Exception:
                    continue
        if hp:
            pl_module.log_metrics(hp, prefix="train/hyperparams")
    
    
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
