"""Evaluation utilities."""

import random
import numpy as np
from utils.rollouts import collect_rollouts, group_trajectories_by_episode
from tsilva_notebook_utils.gymnasium import render_episode_frames


def evaluate_agent(agent, build_env_fn, n_episodes=8, deterministic=True, render=True, 
                  grid=(2, 2), text_color=(0, 0, 0), out_dir="./tmp"):
    """Evaluate agent performance and optionally render episodes."""
    
    # CRITICAL FIX: Force single environment for evaluation
    # 
    # PROBLEM: The same issue that affected training evaluation also affects this function.
    # When build_env_fn uses n_envs="auto", it creates multiple parallel environments.
    # collect_rollouts(n_episodes=8) with multiple environments would collect 8 episodes
    # TOTAL across all environments, not 8 episodes per environment. This caused:
    # 
    # 1. INCOMPLETE EPISODES: Some environments would have partial episodes
    # 2. REWARD CALCULATION ERRORS: Episodes would be incorrectly grouped and rewards
    #    summed across incomplete trajectories
    # 3. INCONSISTENT RESULTS: Evaluation results would vary based on CPU core count
    #    rather than actual agent performance
    # 
    # FIX: Force n_envs=1 to ensure we get exactly n_episodes complete episodes
    # with proper reward calculation and consistent evaluation results.
    trajectories_with_info = collect_rollouts(
        build_env_fn(random.randint(0, 1_000_000), n_envs=1),  # Force single environment
        agent.policy_model,
        n_episodes=n_episodes,
        deterministic=deterministic,
        collect_frames=render
    )
    
    trajectories, info = trajectories_with_info
    
    if render:
        # Extract frames separately (last element in trajectories tuple)
        frames_flat = trajectories[-1]  # frames_env_major from collect_rollouts
        # Get just the first 8 elements for episode grouping (without frames)
        trajectories_no_frames = trajectories[:-1]
    else:
        trajectories_no_frames = trajectories
        frames_flat = None
    
    # Group trajectories by episode
    episodes = group_trajectories_by_episode(trajectories_no_frames)
    mean_reward = np.mean([sum(step[2] for step in episode) for episode in episodes])
    
    results = {
        "mean_reward": mean_reward,
        "episodes": episodes,
        "info": info
    }
    
    if render and frames_flat:
        # Reconstruct episode frames from flat frames
        # frames_flat is in env-major order: [env0_t0, env0_t1, ..., env1_t0, env1_t1, ...]
        # We need to group them by episode using the done signals
        episode_frames = []
        frame_idx = 0
        
        for episode in episodes:
            episode_len = len(episode)
            episode_frames.append(frames_flat[frame_idx:frame_idx + episode_len])
            frame_idx += episode_len
        
        # Render episode frames
        render_episode_frames(episode_frames, out_dir=out_dir, grid=grid, text_color=text_color)
        results["episode_frames"] = episode_frames
    
    return results
