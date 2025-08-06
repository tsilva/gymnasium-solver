#!/usr/bin/env python3
"""
Hyperparameter Control Utility

A command-line tool for adjusting hyperparameters during training.
This script helps you modify hyperparameters on the fly without editing JSON files manually.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def find_latest_run_dir() -> Optional[Path]:
    """Find the latest run directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    # Check for latest-run symlink first
    latest_run = runs_dir / "latest-run"
    if latest_run.exists() and latest_run.is_symlink():
        return latest_run.resolve()
    
    # Fallback: find most recent directory
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name != "latest-run"]
    if not run_dirs:
        return None
    
    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def get_control_file(run_dir: Optional[str] = None) -> Path:
    """Get the hyperparameter control file path."""
    if run_dir:
        control_file = Path(run_dir) / "hyperparam_control" / "hyperparameters.json"
    else:
        latest_run = find_latest_run_dir()
        if latest_run is None:
            raise FileNotFoundError("No run directory found. Please specify --run-dir or start a training run.")
        control_file = latest_run / "hyperparam_control" / "hyperparameters.json"
    
    if not control_file.exists():
        raise FileNotFoundError(f"Control file not found: {control_file}")
    
    return control_file


def load_control_file(file_path: Path) -> Dict[str, Any]:
    """Load the control file safely."""
    if not file_path.exists():
        raise FileNotFoundError(f"Control file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def save_control_file(file_path: Path, data: Dict[str, Any]) -> None:
    """Save the control file safely."""
    data["last_modified"] = time.time()
    
    # Create backup
    backup_path = file_path.with_suffix(f".{int(time.time())}.bak")
    if file_path.exists():
        file_path.rename(backup_path)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {file_path.name}")
        
        # Clean up backup
        if backup_path.exists():
            backup_path.unlink()
            
    except Exception as e:
        # Restore backup on error
        if backup_path.exists():
            backup_path.rename(file_path)
        raise e


def adjust_learning_rate(run_dir: Optional[str], new_lr: float) -> None:
    """Adjust the learning rate."""
    control_file = get_control_file(run_dir)
    
    data = load_control_file(control_file)
    old_lr = data.get("learning_rate", "unknown")
    data["learning_rate"] = new_lr
    
    save_control_file(control_file, data)
    print(f"📈 Learning rate: {old_lr} → {new_lr}")


def adjust_hyperparams(run_dir: Optional[str], **kwargs) -> None:
    """Adjust general hyperparameters."""
    control_file = get_control_file(run_dir)
    
    data = load_control_file(control_file)
    changes = []
    
    for key, value in kwargs.items():
        if value is not None:
            old_value = data.get(key, "not set")
            data[key] = value
            changes.append(f"{key}: {old_value} → {value}")
    
    if changes:
        save_control_file(control_file, data)
        print(f"🎛️  Updated: {', '.join(changes)}")
    else:
        print("ℹ️  No changes specified")


def show_status(run_dir: Optional[str]) -> None:
    """Show current hyperparameter values."""
    control_file = get_control_file(run_dir)
    
    print(f"\n📊 Current Hyperparameters ({control_file})")
    print("=" * 50)
    
    try:
        data = load_control_file(control_file)
        
        # Core hyperparameters
        print(f"Learning Rate: {data.get('learning_rate', 'unknown')}")
        print(f"Entropy Coefficient: {data.get('ent_coef', 'unknown')}")
        print(f"Max Grad Norm: {data.get('max_grad_norm', 'unknown')}")
        
        # Algorithm-specific parameters
        if 'clip_range' in data:
            print(f"Clip Range: {data.get('clip_range', 'unknown')}")
        if 'vf_coef' in data:
            print(f"Value Function Coef: {data.get('vf_coef', 'unknown')}")
        
        # Schedule configuration
        schedule = data.get('schedule', {})
        if schedule:
            print(f"\nSchedule Configuration:")
            print(f"  Adaptive LR: {schedule.get('enable_adaptive_lr', 'unknown')}")
            print(f"  LR Schedule: {schedule.get('lr_schedule', 'unknown')}")
            print(f"  Plateau Patience: {schedule.get('plateau_patience', 'unknown')}")
            print(f"  Plateau Factor: {schedule.get('plateau_factor', 'unknown')}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    
    print()


def configure_schedule(run_dir: Optional[str], 
                      enable_adaptive: Optional[bool] = None,
                      schedule_type: Optional[str] = None,
                      plateau_patience: Optional[int] = None,
                      plateau_factor: Optional[float] = None) -> None:
    """Configure automatic learning rate scheduling."""
    control_file = get_control_file(run_dir)
    
    data = load_control_file(control_file)
    
    # Ensure schedule section exists
    if 'schedule' not in data:
        data['schedule'] = {}
    
    schedule = data['schedule']
    changes = []
    
    if enable_adaptive is not None:
        old_val = schedule.get("enable_adaptive_lr", "unknown")
        schedule["enable_adaptive_lr"] = enable_adaptive
        changes.append(f"adaptive_lr: {old_val} → {enable_adaptive}")
    
    if schedule_type is not None:
        old_val = schedule.get("lr_schedule", "unknown")
        schedule["lr_schedule"] = schedule_type
        changes.append(f"schedule: {old_val} → {schedule_type}")
    
    if plateau_patience is not None:
        old_val = schedule.get("plateau_patience", "unknown")
        schedule["plateau_patience"] = plateau_patience
        changes.append(f"patience: {old_val} → {plateau_patience}")
    
    if plateau_factor is not None:
        old_val = schedule.get("plateau_factor", "unknown")
        schedule["plateau_factor"] = plateau_factor
        changes.append(f"factor: {old_val} → {plateau_factor}")
    
    if changes:
        save_control_file(control_file, data)
        print(f"📅 Schedule updated: {', '.join(changes)}")
    else:
        print("ℹ️  No schedule changes specified")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Control Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                           # Show current values
  %(prog)s lr 0.001                         # Set learning rate to 0.001
  %(prog)s set --ent-coef 0.02              # Set entropy coefficient
  %(prog)s set --ent-coef 0.02 --clip-range 0.1  # Set multiple values
  %(prog)s schedule --enable                # Enable adaptive LR
  %(prog)s schedule --disable --patience 15 # Configure schedule

All changes are made to a single hyperparameters.json file in the run directory.
        """
    )
    
    parser.add_argument("--run-dir", type=str, help="Run directory (default: latest run)")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Status command
    subparsers.add_parser("status", help="Show current hyperparameter values")
    
    # Learning rate command
    lr_parser = subparsers.add_parser("lr", help="Set learning rate")
    lr_parser.add_argument("learning_rate", type=float, help="New learning rate")
    
    # Set hyperparameters command
    set_parser = subparsers.add_parser("set", help="Set hyperparameters")
    set_parser.add_argument("--ent-coef", type=float, help="Entropy coefficient")
    set_parser.add_argument("--max-grad-norm", type=float, help="Maximum gradient norm")
    set_parser.add_argument("--clip-range", type=float, help="PPO clip range")
    set_parser.add_argument("--vf-coef", type=float, help="Value function coefficient")
    
    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Configure learning rate scheduling")
    schedule_group = schedule_parser.add_mutually_exclusive_group()
    schedule_group.add_argument("--enable", action="store_true", help="Enable adaptive LR")
    schedule_group.add_argument("--disable", action="store_false", dest="enable", help="Disable adaptive LR")
    schedule_parser.add_argument("--type", choices=["plateau", "linear", "exponential", "cosine"], 
                                help="Schedule type")
    schedule_parser.add_argument("--patience", type=int, help="Plateau patience (epochs)")
    schedule_parser.add_argument("--factor", type=float, help="LR reduction factor")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == "status":
            show_status(args.run_dir)
        
        elif args.command == "lr":
            adjust_learning_rate(args.run_dir, args.learning_rate)
        
        elif args.command == "set":
            adjust_hyperparams(
                args.run_dir,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
                clip_range=args.clip_range,
                vf_coef=args.vf_coef
            )
        
        elif args.command == "schedule":
            enable_adaptive = args.enable if hasattr(args, 'enable') else None
            configure_schedule(
                args.run_dir,
                enable_adaptive=enable_adaptive,
                schedule_type=args.type,
                plateau_patience=args.patience,
                plateau_factor=args.factor
            )
    
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
