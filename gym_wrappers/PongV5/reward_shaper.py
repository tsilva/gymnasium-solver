import gymnasium as gym
import numpy as np


class PongV5_RewardShaper(gym.Wrapper):
    """
    Reward shaping wrapper for Pong-v5 to encourage paddle-ball interactions.
    
    The original Pong has sparse rewards:
    - +1 for scoring a point
    - -1 for opponent scoring
    - 0 otherwise
    
    This wrapper adds dense shaping rewards based on:
    1. Paddle hits ball (both player and enemy)
    2. Ball proximity to player paddle
    3. Ball direction changes (indicating collisions)
    
    The shaping rewards are designed to encourage active play and ball contact.
    """
    
    def __init__(self, env, paddle_hit_reward=1.0, proximity_reward_scale=0.1, direction_change_reward=0.5):
        super().__init__(env)
        self.paddle_hit_reward = paddle_hit_reward
        self.proximity_reward_scale = proximity_reward_scale
        self.direction_change_reward = direction_change_reward
        
        # Previous state for detecting changes
        self.prev_ball_dx = None
        self.prev_ball_x = None
        self.prev_ball_y = None
        self.prev_player_y = None
        self.prev_enemy_y = None
        
        # Constants for collision detection
        self.paddle_width = 8  # Approximate paddle width in pixels
        self.player_x = 16     # Player paddle X position (left side)
        self.enemy_x = 140     # Enemy paddle X position (right side)
        
    def _detect_paddle_collision(self, ball_x, ball_y, ball_dx, player_y, enemy_y):
        """
        Detect if the ball has collided with either paddle.
        Returns (player_hit, enemy_hit)
        """
        player_hit = False
        enemy_hit = False
        
        # Check player paddle collision (left side)
        if (ball_x <= self.player_x + self.paddle_width and 
            ball_dx > 0 and  # Ball moving right after hitting left paddle
            self.prev_ball_dx is not None and self.prev_ball_dx <= 0):  # Was moving left before
            
            # Check if ball Y is within paddle range
            paddle_top = player_y - 16  # Approximate paddle height
            paddle_bottom = player_y + 16
            if paddle_top <= ball_y <= paddle_bottom:
                player_hit = True
        
        # Check enemy paddle collision (right side)
        if (ball_x >= self.enemy_x - self.paddle_width and 
            ball_dx < 0 and  # Ball moving left after hitting right paddle
            self.prev_ball_dx is not None and self.prev_ball_dx >= 0):  # Was moving right before
            
            # Check if ball Y is within paddle range
            paddle_top = enemy_y - 16  # Approximate paddle height
            paddle_bottom = enemy_y + 16
            if paddle_top <= ball_y <= paddle_bottom:
                enemy_hit = True
                
        return player_hit, enemy_hit
    
    def _calculate_proximity_reward(self, ball_x, ball_y, player_y):
        """
        Calculate reward based on ball proximity to player paddle.
        Encourages the player to keep the paddle near the ball.
        """
        if ball_x > 78:  # Ball is on the right half, no proximity reward
            return 0.0
            
        # Distance between ball and player paddle
        y_distance = abs(ball_y - player_y)
        
        # Reward inversely proportional to distance (closer = better)
        max_distance = 100  # Approximate screen height / 2
        proximity_reward = max(0, (max_distance - y_distance) / max_distance)
        
        return proximity_reward * self.proximity_reward_scale
    
    def _get_ball_state(self):
        """Extract ball and paddle positions from the environment objects."""
        if not hasattr(self.env, 'objects') or not self.env.objects:
            return None, None, None, None, None
            
        obj_map = {}
        for o in self.env.objects:
            if not getattr(o, "hud", False):  # Skip HUD elements
                obj_map[o.category] = o
        
        # Get ball state
        ball = obj_map.get("Ball", None)
        if ball is None:
            return None, None, None, None, None
            
        # Get paddle states
        player = obj_map.get("Player", None)
        enemy = obj_map.get("Enemy", None)
        
        ball_x, ball_y, ball_dx = ball.x, ball.y, ball.dx
        player_y = player.y if player else 0
        enemy_y = enemy.y if enemy else 0
        
        return ball_x, ball_y, ball_dx, player_y, enemy_y
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize previous state
        ball_x, ball_y, ball_dx, player_y, enemy_y = self._get_ball_state()
        self.prev_ball_dx = ball_dx
        self.prev_ball_x = ball_x
        self.prev_ball_y = ball_y
        self.prev_player_y = player_y
        self.prev_enemy_y = enemy_y
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current ball and paddle state
        ball_x, ball_y, ball_dx, player_y, enemy_y = self._get_ball_state()
        
        # Calculate shaping rewards
        shaping_reward = 0.0
        
        if (ball_x is not None and self.prev_ball_dx is not None):
            # 1. Detect paddle collisions
            player_hit, enemy_hit = self._detect_paddle_collision(
                ball_x, ball_y, ball_dx, player_y, enemy_y
            )
            
            collision_reward = 0.0
            if player_hit:
                collision_reward += self.paddle_hit_reward
                info['player_paddle_hit'] = True
            if enemy_hit:
                collision_reward += self.paddle_hit_reward * 0.5  # Less reward for enemy hits
                info['enemy_paddle_hit'] = True
                
            # 2. Proximity reward for keeping paddle near ball
            proximity_reward = self._calculate_proximity_reward(ball_x, ball_y, player_y)
            
            # 3. Direction change reward (indicates any collision, including walls)
            direction_change_reward = 0.0
            if (self.prev_ball_dx is not None and 
                np.sign(ball_dx) != np.sign(self.prev_ball_dx) and 
                np.sign(self.prev_ball_dx) != 0):  # Avoid division by zero
                direction_change_reward = self.direction_change_reward
                info['ball_direction_changed'] = True
            
            shaping_reward = collision_reward + proximity_reward + direction_change_reward
            
            # Add debug info
            info['shaping_reward'] = shaping_reward
            info['collision_reward'] = collision_reward
            info['proximity_reward'] = proximity_reward
            info['direction_change_reward'] = direction_change_reward
        
        # Update previous state
        self.prev_ball_dx = ball_dx
        self.prev_ball_x = ball_x
        self.prev_ball_y = ball_y
        self.prev_player_y = player_y
        self.prev_enemy_y = enemy_y
        
        # Add shaping reward to original reward
        shaped_reward = reward + shaping_reward
        
        return obs, shaped_reward, terminated, truncated, info
