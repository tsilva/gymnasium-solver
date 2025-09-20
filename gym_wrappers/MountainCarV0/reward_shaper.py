import gymnasium as gym
import numpy as np


class MountainCarV0_RewardShaper(gym.Wrapper):
    """Potential-based dense rewards for MountainCar-v0 (position, velocity, height)."""
    
    def __init__(self, env, position_reward_scale=100.0, velocity_reward_scale=10.0, height_reward_scale=50.0):
        super().__init__(env)
        self.position_reward_scale = position_reward_scale
        self.velocity_reward_scale = velocity_reward_scale  
        self.height_reward_scale = height_reward_scale
        
        # MountainCar bounds
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        
        # Previous state for computing deltas
        self.prev_position = None
        self.prev_velocity = None
        self.prev_height = None
        
    def _get_height(self, position):
        """Calculate height on the mountain based on position."""
        # Mountain Car height function: sin(3 * position)
        return np.sin(3 * position)
    
    def _get_position_potential(self, position):
        """Position-based potential: closer to goal = higher potential."""
        # Normalize position to [0, 1] where 1 is at goal
        normalized_pos = (position - self.min_position) / (self.goal_position - self.min_position)
        return normalized_pos
    
    def _get_velocity_potential(self, velocity):
        """Velocity-based potential: positive velocity = higher potential.""" 
        # Normalize velocity to [0, 1] where 1 is max positive velocity
        normalized_vel = (velocity - self.min_velocity) / (self.max_velocity - self.min_velocity)
        return normalized_vel
    
    def _get_height_potential(self, height):
        """Height-based potential: higher on mountain = higher potential."""
        # Height ranges from -1 to 1, normalize to [0, 1]
        return (height + 1) / 2
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        position, velocity = obs
        
        # Initialize previous state
        self.prev_position = position
        self.prev_velocity = velocity  
        self.prev_height = self._get_height(position)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        position, velocity = obs
        height = self._get_height(position)
        
        # Calculate potential-based shaping rewards
        shaping_reward = 0.0
        
        if self.prev_position is not None:
            # Position progress reward (potential difference)
            curr_pos_potential = self._get_position_potential(position)
            prev_pos_potential = self._get_position_potential(self.prev_position)
            position_shaping = self.position_reward_scale * (curr_pos_potential - prev_pos_potential)
            
            # Velocity shaping reward (encourage positive velocity)
            curr_vel_potential = self._get_velocity_potential(velocity)
            prev_vel_potential = self._get_velocity_potential(self.prev_velocity)
            velocity_shaping = self.velocity_reward_scale * (curr_vel_potential - prev_vel_potential)
            
            # Height progress reward (potential difference)
            curr_height_potential = self._get_height_potential(height)
            prev_height_potential = self._get_height_potential(self.prev_height)
            height_shaping = self.height_reward_scale * (curr_height_potential - prev_height_potential)
            
            shaping_reward = position_shaping + velocity_shaping + height_shaping
            
            # Add debug info
            info['shaping_reward'] = shaping_reward
            info['position_shaping'] = position_shaping
            info['velocity_shaping'] = velocity_shaping
            info['height_shaping'] = height_shaping
        
        # Update previous state
        self.prev_position = position
        self.prev_velocity = velocity
        self.prev_height = height
        
        # Add shaping reward to original reward
        shaped_reward = reward + shaping_reward
        
        return obs, shaped_reward, terminated, truncated, info
