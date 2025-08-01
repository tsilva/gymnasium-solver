#!/usr/bin/env python3
"""
Play script to visualize a trained RL agent.

This script loads a saved model and runs it in the environment with human-readable rendering,
allowing you to see the trained policy in action.
"""

import argparse
import torch
import time
import numpy as np
from pathlib import Path


def load_model(model_path, config):
    """Load a saved model and return the policy."""
    from agents import create_agent
    
    # Create agent with the same config
    agent = create_agent(config)
    
    # Load the saved state dict with weights_only=False since we trust our own files
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent.policy_model.load_state_dict(checkpoint['model_state_dict'])
    agent.policy_model.eval()  # Set to evaluation mode
    
    print(f"Loaded model from {model_path}")
    print(f"Model was trained for {checkpoint.get('total_timesteps', 'unknown')} timesteps")
    print(f"Best eval reward: {checkpoint.get('eval_reward', 'unknown')}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Handle backward compatibility - if config is saved as object, it will be in 'config' key
    # If saved as dict, it will be in 'config_dict' key
    if 'config_dict' in checkpoint:
        saved_config = checkpoint['config_dict']
        print(f"Loaded config from checkpoint (dict format)")
    elif 'config' in checkpoint:
        saved_config = checkpoint['config']
        print(f"Loaded config from checkpoint (object format)")
    else:
        print("No config found in checkpoint, using provided config")
    
    return agent.policy_model


def play_episodes(policy_model, env, num_episodes=5, deterministic=True):
    """Run episodes with the trained policy and render them."""
    
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
            
            # Render the environment
            env.render()
            
            # Small delay to make it watchable
            time.sleep(0.05)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1} finished after {step_count} steps with reward: {episode_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Play trained RL agent.")
    parser.add_argument("--config", type=str, default="CartPole-v1", 
                       help="Config ID (should match training config)")
    parser.add_argument("--algo", type=str, default="ppo", 
                       help="Algorithm used for training")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to saved model (if not provided, auto-detect)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to play")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic policy (default: deterministic)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load config
    from utils.config import load_config
    config = load_config(args.config, args.algo)
    
    # Set random seed
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(args.seed)
    
    # Determine model path
    if args.model is None:
        # Auto-detect model path
        model_filename = f"best_model_{config.env_id.replace('/', '_')}_{config.algo_id}.pth"
        model_path = Path("saved_models") / model_filename
        
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            print("Available models:")
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
    
    # Create environment with human rendering
    from utils.environment import build_env
    env = build_env(
        config.env_id,
        seed=args.seed,
        env_wrappers=config.env_wrappers,
        norm_obs=config.normalize_obs,
        n_envs=1,  # Single environment for playing
        frame_stack=config.frame_stack,
        obs_type=config.obs_type,
        render_mode="human"  # Human-readable rendering
    )
    
    print(f"\nEnvironment: {config.env_id}")
    print(f"Algorithm: {config.algo_id}")
    print(f"Deterministic: {not args.stochastic}")
    print(f"Episodes to play: {args.episodes}")
    
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
