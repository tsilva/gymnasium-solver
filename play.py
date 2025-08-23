#!/usr/bin/env python3
"""
Play script to visualize a trained RL agent.

This script loads a saved model and runs it in the environment with human-readable rendering,
allowing you to see the trained policy in action.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import os
import time
import platform


def load_model(model_path, config):
    """Load a saved model and return the policy.

    This function avoids constructing a full training Agent to prevent any
    side effects (like creating run directories or symlinks). It infers the
    policy architecture from the config and environment spec, builds the
    appropriate policy network, and loads the checkpoint weights.
    """

    # Load the saved state dict with weights_only=False since we trust our own files
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Infer model architecture from config and a lightweight env spec
    from utils.environment import build_env
    from utils.policy_factory import create_actor_critic_policy, create_policy

    # Build a minimal env to infer input/output dimensions and observation space
    env = build_env(
        config.env_id,
        seed=42,
        env_wrappers=config.env_wrappers,
        norm_obs=config.normalize_obs,
        n_envs=1,
        frame_stack=config.frame_stack,
        obs_type=config.obs_type,
        render_mode=None,
        env_kwargs=config.env_kwargs,
        subproc=False,
        grayscale_obs=getattr(config, "grayscale_obs", False),
        resize_obs=getattr(config, "resize_obs", False),
    )
    try:
        # VecInfoWrapper exposes helpers for dims; also keep obs_space for CNN policies
        input_dim = env.get_input_dim()
        output_dim = env.get_output_dim()
        obs_space = getattr(env, "observation_space", None)

        # Policy selection based on algo_id (policy-only for REINFORCE)
        policy_type = getattr(config, "policy", "mlp")
        policy_kwargs = dict(getattr(config, "policy_kwargs", None) or {})
        activation = policy_kwargs.pop("activation", getattr(config, "activation", "tanh"))

        if str(getattr(config, "algo_id", "")).lower() == "reinforce":
            model = create_policy(
                policy_type,
                input_dim=int(input_dim),
                action_dim=int(output_dim),
                hidden=config.hidden_dims,
                activation=activation,
                obs_space=obs_space,
                **policy_kwargs,
            )
        else:
            model = create_actor_critic_policy(
                policy_type,
                input_dim=int(input_dim),
                action_dim=int(output_dim),
                hidden=config.hidden_dims,
                activation=activation,
                obs_space=obs_space,
                **policy_kwargs,
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Check if this is a checkpoint format or legacy format
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume legacy format where checkpoint IS the state dict
        model.load_state_dict(checkpoint)

    model.eval()  # Set to evaluation mode

    print(f"Loaded model from {model_path}")
    print(f"Model was trained for {checkpoint.get('total_timesteps', 'unknown')} timesteps")

    # Handle different checkpoint formats (best-effort informational prints)
    if 'best_eval_reward' in checkpoint:
        print(f"Best eval reward: {checkpoint.get('best_eval_reward', 'unknown')}")
        if 'current_eval_reward' in checkpoint:
            print(f"Current eval reward: {checkpoint.get('current_eval_reward', 'unknown')}")
        print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
        flags = []
        if checkpoint.get('is_best', False):
            flags.append("best")
        if checkpoint.get('is_last', False):
            flags.append("last")
        if checkpoint.get('is_threshold', False):
            flags.append("threshold")
        if flags:
            print(f"Checkpoint type: {', '.join(flags)}")
    else:
        print(f"Best eval reward: {checkpoint.get('eval_reward', 'unknown')}")
        print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")

    # Handle backward compatibility - if config is saved as object, it will be in 'config' key
    # If saved as dict, it will be in 'config_dict' key
    if 'config_dict' in checkpoint:
        _ = checkpoint['config_dict']
        print(f"Loaded config from checkpoint (dict format)")
    elif 'config' in checkpoint:
        _ = checkpoint['config']
        print(f"Loaded config from checkpoint (object format)")
    else:
        print("No config found in checkpoint, using provided config")

    return model


def play_episodes(policy_model, env, num_episodes=5, deterministic=False):
    """Run episodes with the trained policy and render them."""
    # Infer FPS for pacing playback
    fps = None
    try:
        if hasattr(env, "get_render_fps"):
            fps = env.get_render_fps()  # type: ignore[attr-defined]
    except Exception:
        fps = None
    if not isinstance(fps, int) or fps <= 0:
        try:
            # Try reading metadata from first underlying env
            md = None
            if hasattr(env, "venv") and hasattr(env.venv, "get_attr"):
                lst = env.venv.get_attr("metadata", indices=[0])
                if isinstance(lst, list) and lst:
                    md = lst[0]
            elif hasattr(env, "metadata"):
                md = getattr(env, "metadata", None)
            if isinstance(md, dict):
                rfps = md.get("render_fps")
                if isinstance(rfps, (int, float)) and rfps > 0:
                    fps = int(rfps)
        except Exception:
            fps = None
    frame_interval = (1.0 / float(fps)) if isinstance(fps, int) and fps > 0 else None

    for episode in range(num_episodes):
        # Handle both new and old gymnasium reset API
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
            
        episode_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        # Track time to pace frames close to the environment FPS
        last_time = time.perf_counter()
        while True:
            # Get action from policy
            with torch.no_grad():
                # Handle vectorized environments - obs might be an array
                if isinstance(obs, np.ndarray) and obs.ndim > 1:
                    obs_tensor = torch.FloatTensor(obs)
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    
                dist, value = policy_model(obs_tensor)
                
                if deterministic:
                    # Use mean action for deterministic policy
                    if hasattr(dist, 'mode'):
                        action = dist.mode
                    else:
                        action = dist.mean
                else:
                    # Sample from the distribution
                    action = dist.sample()
                
                action = action.squeeze().cpu().numpy()
                
                # Handle discrete vs continuous action spaces
                if len(action.shape) == 0:  # Scalar (discrete)
                    action = int(action.item())
                elif action.shape == (1,):  # Single continuous action
                    action = action.item()
                
                # For vectorized environments, we need to wrap single actions in array
                if hasattr(env, 'num_envs') and env.num_envs == 1:
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])
            
            # Take step in environment
            step_result = env.step(action)
            
            # Handle different step return formats
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                raise ValueError(f"Unexpected step result length: {len(step_result)}")
            
            # Handle vectorized rewards/dones
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0]
            if isinstance(terminated, (list, np.ndarray)):
                terminated = terminated[0]
            if isinstance(truncated, (list, np.ndarray)):
                truncated = truncated[0]
            
            episode_reward += reward
            step_count += 1
            
            # Always attempt to render; some vectorized wrappers don't expose render_mode
            try:
                env.render()
            except Exception:
                pass

            # Pace playback to match FPS if known
            if frame_interval is not None:
                try:
                    now = time.perf_counter()
                    sleep_s = frame_interval - (now - last_time)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    last_time = time.perf_counter()
                except Exception:
                    pass
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} finished after {step_count} steps with reward: {episode_reward:.2f}")


def load_config_from_run(run_id: str):
    """Load configuration from a run's config.json file."""
    import json

    from utils.config import Config
    
    # Handle latest-run symlink
    runs_dir = Path("runs")
    if run_id == "latest-run":
        run_path = runs_dir / "latest-run"
        if run_path.is_symlink():
            run_id = str(run_path.readlink())
        else:
            raise FileNotFoundError("latest-run symlink not found")
    
    run_path = runs_dir / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    # Prefer new layout at run root; fallback to legacy configs/config.json
    config_file = run_path / "config.json"
    if not config_file.exists():
        config_file = run_path / "configs" / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    # Create Config instance from pldictionary
    config = Config(**config_dict)
    return config


def find_best_checkpoint_in_run(run_id: str) -> Path:
    """Find the best checkpoint in a run directory."""
    runs_dir = Path("runs")
    
    # Handle latest-run symlink
    if run_id == "latest-run":
        run_path = runs_dir / "latest-run"
        if run_path.is_symlink():
            run_id = str(run_path.readlink())
        else:
            raise FileNotFoundError("latest-run symlink not found")
    
    run_path = runs_dir / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    
    checkpoints_dir = run_path / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Look for best checkpoint first (new and legacy names)
    best_checkpoint = checkpoints_dir / "best.ckpt"
    if not best_checkpoint.exists():
        best_checkpoint = checkpoints_dir / "best_checkpoint.ckpt"
    if best_checkpoint.exists():
        return best_checkpoint
    
    # Fall back to last checkpoint (new and legacy names)
    last_checkpoint = checkpoints_dir / "last.ckpt"
    if not last_checkpoint.exists():
        last_checkpoint = checkpoints_dir / "last_checkpoint.ckpt"
    if last_checkpoint.exists():
        return last_checkpoint
    
    # Look for any checkpoint files
    checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
    if checkpoint_files:
        # Sort by modification time, return most recent
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoint_files[0]
    
    raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")


def main():
    parser = argparse.ArgumentParser(description="Play trained RL agent.")
    parser.add_argument("--run-id", type=str, default="latest-run",
                       help="Run ID to load model from (default: latest-run)")
    parser.add_argument("--config", type=str, default=None, 
                       help="Config ID (if not provided, load from run)")
    parser.add_argument("--algo", type=str, default=None, 
                       help="Algorithm used for training (if not provided, load from run)")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to saved model (if not provided, auto-detect from run)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to play")
    parser.add_argument("--stochastic", #action="store_true",
                       default=True, help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    parser.add_argument("--render", type=str, default="auto", choices=["auto", "human", "rgb_array", "none"],
                       help="Rendering mode: auto (default), human, rgb_array, or none")
    
    args = parser.parse_args()
    
    # Load config - either from run or from provided config/algo
    if args.config is not None and args.algo is not None:
        # Legacy mode: load config from config files
        from utils.config import load_config
        config = load_config(args.config, args.algo)
        print(f"Loaded config from files: {args.config}/{args.algo}")
    else:
        # New mode: load config from run
        try:
            config = load_config_from_run(args.run_id)
            print(f"Loaded config from run: {args.run_id}")
            print(f"  Environment: {config.env_id}")
            print(f"  Algorithm: {config.algo_id}")
        except FileNotFoundError as e:
            print(f"Error loading config from run: {e}")
            if args.config is None or args.algo is None:
                print("Please provide either --run-id (with config in run) or both --config and --algo")
                return
            # Fall back to legacy mode
            from utils.config import load_config
            config = load_config(args.config, args.algo)
            print(f"Falling back to config from files: {args.config}/{args.algo}")
    
    # Set up logging for play session
    from utils.logging import capture_all_output, log_config_details
    
    with capture_all_output(config=config, log_dir=args.log_dir):
        print(f"=== Play Session Started ===")
        print(f"Command: {' '.join(['python', 'play.py'] + ['--run-id', args.run_id] + (['--config', args.config] if args.config else []) + (['--algo', args.algo] if args.algo else []) + (['--model', args.model] if args.model else []) + (['--stochastic'] if args.stochastic else []))}")
        # Log configuration for the play session as well
        try:
            log_config_details(config)
        except Exception:
            pass
        
        # Set random seed
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(args.seed)
        
        # Determine model path
        if args.model is None:
            # Auto-detect model path from run directory first
            try:
                model_path = find_best_checkpoint_in_run(args.run_id)
                print(f"Found checkpoint in run {args.run_id}: {model_path}")
            except FileNotFoundError as e:
                print(f"No checkpoint found in run: {e}")
                # Fall back to old checkpoint system
                from utils.checkpoint import (
                    find_latest_checkpoint,
                    list_available_checkpoints,
                )
                
                checkpoint_path = find_latest_checkpoint(config.algo_id, config.env_id)
                
                if checkpoint_path:
                    model_path = checkpoint_path
                    print(f"Found checkpoint: {model_path}")
                else:
                    # Fall back to old saved_models directory
                    model_filename = f"best_model_{config.env_id.replace('/', '_')}_{config.algo_id}.pth"
                    model_path = Path("saved_models") / model_filename
                    
                    if not model_path.exists():
                        print(f"No model found for {config.algo_id}/{config.env_id}")
                        print("\nAvailable checkpoints:")
                        checkpoints = list_available_checkpoints()
                        if checkpoints:
                            for algo, envs in checkpoints.items():
                                for env, files in envs.items():
                                    print(f"  {algo}/{env}: {files}")
                        else:
                            print("  No checkpoints found")
                        
                        print("\nAvailable legacy models:")
                        models_dir = Path("saved_models")
                        if models_dir.exists():
                            for model_file in models_dir.glob("*.pth"):
                                print(f"  {model_file}")
                        else:
                            print("  No saved_models directory found")
                        return
        else:
            model_path = Path(args.model)
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return
        
        # Load the trained model
        policy_model = load_model(model_path, config)
        
        # Decide render mode (auto-detect headless/WSL)
        def _choose_render_mode():
            if args.render in {"human", "rgb_array", "none"}:
                return None if args.render == "none" else args.render
            # auto
            is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
            has_display = bool(
                os.environ.get("DISPLAY")
                or os.environ.get("WAYLAND_DISPLAY")
                or os.environ.get("MIR_SOCKET")
            )
            # Prefer human rendering whenever a display is available, even on WSL/WSLg
            if has_display:
                return "human"
            # Fallback to rgb_array for headless environments
            return "rgb_array"

        chosen_render_mode = _choose_render_mode()

        # Create environment with chosen rendering
        from utils.environment import build_env
        # On WSL, some OpenGL drivers (GLX) may fail due to libstdc++ mismatches from Conda.
        # Prefer SDL software renderer to avoid GLX context creation issues when using human rendering.
        try:
            if chosen_render_mode == "human":
                is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
                if is_wsl:
                    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
        except Exception:
            pass
        env = build_env(
            config.env_id,
            seed=args.seed,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            n_envs=1,  # Force single environment for playing
            frame_stack=config.frame_stack,
            obs_type=config.obs_type,
            render_mode=chosen_render_mode,
            env_kwargs=config.env_kwargs,
            subproc=False,  # Force DummyVecEnv for playing
            # Match training-time preprocessing so model input shapes align
            grayscale_obs=getattr(config, "grayscale_obs", False),
            resize_obs=getattr(config, "resize_obs", False),
        )
        
        # Verify we have a single environment with DummyVecEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        if not isinstance(env.venv, DummyVecEnv):
            print(f"Warning: Expected DummyVecEnv but got {type(env.venv)}")
        if env.num_envs != 1:
            print(f"Warning: Expected 1 environment but got {env.num_envs}")
        else:
            print(f"✓ Using single environment with DummyVecEnv")
        
        print(f"\nEnvironment: {config.env_id}")
        print(f"Algorithm: {config.algo_id}")
        print(f"Deterministic: {not args.stochastic}")
        print(f"Episodes to play: {args.episodes}")
        try:
            rm = getattr(env, "render_mode", chosen_render_mode)
            print(f"Render mode: {rm}")
        except Exception:
            pass
        
        try:
            # Play episodes
            play_episodes(
                policy_model, 
                env, 
                num_episodes=args.episodes,
                deterministic=not args.stochastic
            )
        finally:
            env.close()
        
        print("\nDone playing!")


if __name__ == "__main__":
    main()
