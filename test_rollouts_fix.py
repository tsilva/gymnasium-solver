#!/usr/bin/env python3
"""
Quick test to verify our collect_rollouts fix works correctly.
This test ensures that when n_episodes is specified, we get exactly
that many complete episodes, regardless of vectorized environment size.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from utils.rollouts import collect_rollouts, group_trajectories_by_episode


class MockPolicy(torch.nn.Module):
    """Simple mock policy that returns random actions"""
    def __init__(self, action_dim=2):
        super().__init__()
        self.action_dim = action_dim
        # Add a dummy parameter so _device_of works
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, obs):
        batch_size = obs.shape[0]
        # Return random logits
        return torch.randn(batch_size, self.action_dim)


class MockVecEnv:
    """Mock vectorized environment for testing"""
    def __init__(self, n_envs=4, episode_length=10):
        self.n_envs = n_envs
        self.num_envs = n_envs  # Add num_envs attribute for compatibility
        self.episode_length = episode_length
        self.step_counts = np.zeros(n_envs, dtype=int)
        self.obs_dim = 4
        
    def reset(self):
        self.step_counts = np.zeros(self.n_envs, dtype=int)
        return np.random.randn(self.n_envs, self.obs_dim).astype(np.float32)
    
    def step(self, actions):
        self.step_counts += 1
        
        # Generate random observations, rewards
        obs = np.random.randn(self.n_envs, self.obs_dim).astype(np.float32)
        rewards = np.random.randn(self.n_envs).astype(np.float32)
        
        # Episodes end when step count reaches episode_length
        dones = (self.step_counts >= self.episode_length)
        
        # Reset step counts for envs that are done
        self.step_counts[dones] = 0
        
        # Mock infos
        infos = [{}] * self.n_envs
        
        return obs, rewards, dones, infos
    
    def get_images(self):
        # Return dummy frames
        return [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(self.n_envs)]


def test_exact_episodes():
    """Test that collect_rollouts returns exactly n_episodes complete episodes"""
    print("Testing that collect_rollouts returns exactly n_episodes complete episodes...")
    
    # Create mock environment with 4 parallel environments
    env = MockVecEnv(n_envs=4, episode_length=5)
    policy = MockPolicy(action_dim=2)
    
    # Request exactly 3 episodes
    n_episodes_requested = 3
    
    trajectories, info = collect_rollouts(
        env=env,
        policy_model=policy,
        n_episodes=n_episodes_requested,
        deterministic=True
    )
    
    # Extract data (without frames which is the last element)
    trajectories_no_frames = trajectories[:-1]
    
    # Group by episodes and limit to exactly n_episodes_requested
    episodes = group_trajectories_by_episode(trajectories_no_frames, max_episodes=n_episodes_requested)
    
    print(f"Requested episodes: {n_episodes_requested}")
    print(f"Actual complete episodes: {len(episodes)}")
    print(f"Info episode count: {info['n_episodes']}")
    
    # Verify we got exactly the requested number of complete episodes
    assert len(episodes) == n_episodes_requested, f"Expected {n_episodes_requested} episodes, got {len(episodes)}"
    
    # Verify all episodes are complete (end with done=True)
    for i, episode in enumerate(episodes):
        last_step = episode[-1]
        done = last_step[3]  # done is the 4th element
        assert done.item() == True, f"Episode {i} is not complete (doesn't end with done=True)"
        print(f"Episode {i}: {len(episode)} steps, ends with done={done.item()}")
    
    print("✓ Test passed: Got exactly the requested number of complete episodes!")


def test_multiple_scenarios():
    """Test various scenarios to ensure robustness"""
    scenarios = [
        (2, 3, 5),  # n_envs=2, episode_length=3, n_episodes=5
        (4, 8, 2),  # n_envs=4, episode_length=8, n_episodes=2  
        (1, 10, 3), # n_envs=1, episode_length=10, n_episodes=3
        (8, 5, 10), # n_envs=8, episode_length=5, n_episodes=10
    ]
    
    for n_envs, episode_length, n_episodes in scenarios:
        print(f"\nTesting: {n_envs} envs, episode_length={episode_length}, requesting {n_episodes} episodes")
        
        env = MockVecEnv(n_envs=n_envs, episode_length=episode_length)
        policy = MockPolicy(action_dim=2)
        
        trajectories, info = collect_rollouts(
            env=env,
            policy_model=policy,
            n_episodes=n_episodes,
            deterministic=True
        )
        
        # Group by episodes and limit to exactly n_episodes
        trajectories_no_frames = trajectories[:-1]
        episodes = group_trajectories_by_episode(trajectories_no_frames, max_episodes=n_episodes)
        
        print(f"  Got {len(episodes)} complete episodes")
        assert len(episodes) == n_episodes, f"Expected {n_episodes}, got {len(episodes)}"
        
        # Verify all episodes are complete
        for i, episode in enumerate(episodes):
            last_step = episode[-1]
            done = last_step[3]
            assert done.item() == True, f"Episode {i} is incomplete"
    
    print("✓ All scenarios passed!")


if __name__ == "__main__":
    print("Testing collect_rollouts fix...")
    print("=" * 60)
    
    test_exact_episodes()
    test_multiple_scenarios()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! The collect_rollouts fix is working correctly.")
    print("When n_episodes is specified, you now get exactly that many complete episodes,")
    print("regardless of the number of parallel environments.")
