#!/usr/bin/env python3
"""Checkpoint management utility for the gymnasium-solver project."""

import argparse
import os
import torch
from pathlib import Path
from utils.checkpoint import list_available_checkpoints, find_latest_checkpoint, load_checkpoint
from utils.config import load_config

def list_checkpoints(checkpoint_dir: str = "checkpoints"):
    """List all available checkpoints."""
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Available checkpoints in {checkpoint_dir}:")
    print("=" * 60)
    
    for algo_id, envs in checkpoints.items():
        print(f"\nAlgorithm: {algo_id}")
        print("-" * 40)
        
        for env_id, checkpoint_files in envs.items():
            print(f"  Environment: {env_id}")
            for checkpoint_file in checkpoint_files:
                if checkpoint_file == "best_checkpoint.ckpt":
                    print(f"    ✓ {checkpoint_file} (best)")
                elif checkpoint_file == "last_checkpoint.ckpt":
                    print(f"    → {checkpoint_file} (last)")
                elif checkpoint_file.startswith("threshold-"):
                    print(f"    🎯 {checkpoint_file} (threshold)")
                elif "epoch=" in checkpoint_file and "step=" in checkpoint_file:
                    print(f"    📁 {checkpoint_file} (timestamped)")
                else:
                    print(f"    📄 {checkpoint_file}")


def show_checkpoint_info(algo_id: str, env_id: str, checkpoint_dir: str = "checkpoints"):
    """Show detailed information about the latest checkpoint for an algorithm/environment."""
    checkpoint_path = find_latest_checkpoint(algo_id, env_id, checkpoint_dir)
    
    if not checkpoint_path:
        print(f"No checkpoint found for {algo_id}/{env_id}")
        return
    
    print(f"Latest checkpoint for {algo_id}/{env_id}:")
    print(f"Path: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\nCheckpoint Information:")
        print("-" * 30)
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Global Step: {checkpoint.get('global_step', 'unknown')}")
        print(f"Total Timesteps: {checkpoint.get('total_timesteps', 'unknown')}")
        print(f"Best Eval Reward: {checkpoint.get('best_eval_reward', 'unknown')}")
        print(f"Current Eval Reward: {checkpoint.get('current_eval_reward', 'unknown')}")
        print(f"Is Best: {checkpoint.get('is_best', False)}")
        print(f"Is Last: {checkpoint.get('is_last', False)}")
        print(f"Is Threshold: {checkpoint.get('is_threshold', False)}")
        
        if 'config_dict' in checkpoint:
            config = checkpoint['config_dict']
            print(f"\nTraining Configuration:")
            print("-" * 30)
            print(f"Environment: {config.get('env_id', 'unknown')}")
            print(f"Algorithm: {config.get('algo_id', 'unknown')}")
            print(f"Seed: {config.get('seed', 'unknown')}")
            print(f"Learning Rate: {config.get('policy_lr', 'unknown')}")
            print(f"Batch Size: {config.get('batch_size', 'unknown')}")
            print(f"Max Epochs: {config.get('max_epochs', 'unknown')}")
            if config.get('reward_threshold'):
                print(f"Reward Threshold: {config.get('reward_threshold')}")
    
    except Exception as e:
        print(f"Error reading checkpoint: {e}")


def clean_checkpoints(algo_id: str, env_id: str, checkpoint_dir: str = "checkpoints", 
                     keep_best: bool = True, keep_last: bool = True, keep_threshold: bool = True):
    """Clean old timestamped checkpoints, keeping only specified types."""
    env_id_clean = env_id.replace('/', '_').replace('', '_')
    checkpoint_path = Path(checkpoint_dir) / algo_id / env_id_clean
    
    if not checkpoint_path.exists():
        print(f"No checkpoint directory found for {algo_id}/{env_id}")
        return
    
    # Find all timestamped checkpoints (not the special ones)
    timestamped_checkpoints = []
    for checkpoint_file in checkpoint_path.glob("epoch=*-step=*.ckpt"):
        if not checkpoint_file.name.startswith("threshold-"):
            timestamped_checkpoints.append(checkpoint_file)
    
    if not timestamped_checkpoints:
        print(f"No timestamped checkpoints found to clean for {algo_id}/{env_id}")
        return
    
    print(f"Found {len(timestamped_checkpoints)} timestamped checkpoints for {algo_id}/{env_id}")
    print("Files to be removed:")
    
    for checkpoint_file in timestamped_checkpoints:
        print(f"  {checkpoint_file.name}")
    
    response = input("Are you sure you want to delete these files? (y/N): ")
    if response.lower() in ['y', 'yes']:
        for checkpoint_file in timestamped_checkpoints:
            checkpoint_file.unlink()
            print(f"Deleted: {checkpoint_file.name}")
        print(f"Cleaned {len(timestamped_checkpoints)} timestamped checkpoints")
    else:
        print("Cleanup cancelled")


def main():
    parser = argparse.ArgumentParser(description="Manage checkpoints for gymnasium-solver")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--log", action="store_true", help="Enable logging to file")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all available checkpoints")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed info about a checkpoint")
    info_parser.add_argument("algo_id", help="Algorithm ID (e.g., 'ppo', 'reinforce')")
    info_parser.add_argument("env_id", help="Environment ID (e.g., 'CartPole-v1')")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old timestamped checkpoints")
    clean_parser.add_argument("algo_id", help="Algorithm ID (e.g., 'ppo', 'reinforce')")
    clean_parser.add_argument("env_id", help="Environment ID (e.g., 'CartPole-v1')")
    clean_parser.add_argument("--keep-best", action="store_true", default=True, help="Keep best checkpoint")
    clean_parser.add_argument("--keep-last", action="store_true", default=True, help="Keep last checkpoint")
    clean_parser.add_argument("--keep-threshold", action="store_true", default=True, help="Keep threshold checkpoints")
    
    args = parser.parse_args()
    
    # Set up logging if requested
    if args.log:
        from utils.logging import capture_all_output
        with capture_all_output(log_dir=args.log_dir):
            _execute_command(args)
    else:
        _execute_command(args)


def _execute_command(args):
    """Execute the parsed command."""
    if args.command == "list":
        list_checkpoints(args.checkpoint_dir)
    elif args.command == "info":
        show_checkpoint_info(args.algo_id, args.env_id, args.checkpoint_dir)
    elif args.command == "clean":
        clean_checkpoints(args.algo_id, args.env_id, args.checkpoint_dir,
                         args.keep_best, args.keep_last, args.keep_threshold)
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.print_help()


if __name__ == "__main__":
    main()


def show_checkpoint_details(algo_id, env_id, checkpoint_dir="checkpoints"):
    """Show detailed information about checkpoints for a specific algorithm/environment."""
    checkpoint_path = find_latest_checkpoint(algo_id, env_id, checkpoint_dir)
    
    if not checkpoint_path:
        print(f"No checkpoints found for {algo_id}/{env_id}")
        return
    
    print(f"Latest checkpoint for {algo_id}/{env_id}:")
    print(f"Path: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\nCheckpoint details:")
        print("-" * 40)
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
        print(f"Total timesteps: {checkpoint.get('total_timesteps', 'unknown')}")
        print(f"Best eval reward: {checkpoint.get('best_eval_reward', checkpoint.get('eval_reward', 'unknown'))}")
        print(f"Current eval reward: {checkpoint.get('current_eval_reward', 'unknown')}")
        print(f"Is best: {checkpoint.get('is_best', False)}")
        print(f"Is threshold: {checkpoint.get('is_threshold', False)}")
        print(f"Is last: {checkpoint.get('is_last', False)}")
        
        if 'config_dict' in checkpoint:
            config = checkpoint['config_dict']
            print(f"\nTraining configuration:")
            print(f"  Environment: {config.get('env_id', 'unknown')}")
            print(f"  Algorithm: {config.get('algo_id', 'unknown')}")
            print(f"  Seed: {config.get('seed', 'unknown')}")
            print(f"  Learning rate: {config.get('policy_lr', config.get('learning_rate', 'unknown'))}")
            print(f"  Batch size: {config.get('batch_size', 'unknown')}")
            print(f"  Hidden dims: {config.get('hidden_dims', 'unknown')}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")


def clean_checkpoints(algo_id=None, env_id=None, keep_best=True, keep_threshold=True, checkpoint_dir="checkpoints"):
    """Clean up old checkpoints, optionally keeping best and threshold checkpoints."""
    import shutil
    
    checkpoints = list_available_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print("No checkpoints found to clean.")
        return
    
    for current_algo, envs in checkpoints.items():
        if algo_id and current_algo != algo_id:
            continue
            
        for current_env, files in envs.items():
            if env_id and current_env != env_id:
                continue
            
            checkpoint_dir_path = Path(checkpoint_dir) / current_algo / current_env
            files_to_remove = []
            
            for file in files:
                file_path = checkpoint_dir_path / file
                
                if not file_path.exists():
                    continue
                
                try:
                    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                    is_best = checkpoint.get('is_best', False)
                    is_threshold = checkpoint.get('is_threshold', False)
                    
                    should_keep = False
                    if keep_best and is_best:
                        should_keep = True
                    if keep_threshold and is_threshold:
                        should_keep = True
                    
                    if not should_keep and file == "last_checkpoint.ckpt":
                        # Always consider removing last checkpoint unless it's also best/threshold
                        files_to_remove.append(file_path)
                    elif not should_keep:
                        files_to_remove.append(file_path)
                        
                except Exception as e:
                    print(f"Error checking {file_path}: {e}")
            
            # Remove files
            for file_path in files_to_remove:
                print(f"Removing {file_path}")
                file_path.unlink()
            
            # Remove empty directories
            if not any(checkpoint_dir_path.iterdir()):
                print(f"Removing empty directory {checkpoint_dir_path}")
                checkpoint_dir_path.rmdir()
                
                # Remove parent if empty
                if not any(checkpoint_dir_path.parent.iterdir()):
                    print(f"Removing empty directory {checkpoint_dir_path.parent}")
                    checkpoint_dir_path.parent.rmdir()
