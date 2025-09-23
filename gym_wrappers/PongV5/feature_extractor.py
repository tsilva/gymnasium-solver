import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Screen dimensions for Atari Pong (for normalizing positions)
SCREEN_W: float = 160.0
SCREEN_H: float = 210.0

# Approximate paddle height in pixels (used when not available on objects)
# Note: OCAtari objects may expose size; we prefer that when present.
PADDLE_H_DEFAULT: float = 32.0

# Max Y pixels per frame that paddle can move (for normalizing velocity)
PADDLE_DY_SCALE: float = 24.0

# Max X/Y pixels per frame that ball can move (for normalizing velocity)
BALL_D_SCALE: float = 12.0

# ---- helpers from your snippet ----
def _index_objects_by_category(objects):
    objects_map = {}
    for object in objects:
        # Skip HUD objects
        if getattr(object, "hud", False): continue
       
        # TODO: figure out what causes this
        # Skip objects that already exist in map
        if object.category in objects_map: continue

        # Assert that object category is not already in map
        #assert object.category not in objects_map, f"Object {object.category} already exists in map"

        # Add object to map
        objects_map[object.category] = object

    return objects_map

def _normalize_velocity(value: float, scale: float) -> float:
    """Map symmetric value in R to [0,1] using tanh with scale.

    0 maps to 0.5, extremes saturate to 0/1.
    """
    return 0.5 * (np.tanh(float(value) / float(scale)) + 1.0)

def _paddle_height(obj) -> float:
    """Return paddle height from object if available, else a sensible default."""
    for attr in ("height", "h"):
        if hasattr(obj, attr):
            try:
                return float(getattr(obj, attr))
            except Exception:
                continue
    return PADDLE_H_DEFAULT


def _normalize_paddle_center_y(center_y: float, paddle_h: float) -> float:
    """Normalize paddle center-Y to [0,1] over playable range.

    Maps center_y in [paddle_h/2, SCREEN_H - paddle_h/2] to [0, 1].
    """
    denom = SCREEN_H - float(paddle_h)
    assert denom > 0.0, f"Invalid paddle height {paddle_h} for screen height {SCREEN_H}"
    return (float(center_y) - 0.5 * float(paddle_h)) / denom


def _obs_from_objects(objects):
    """
    Vector order (length=8): [Player.y, Player.dy, Enemy.y, Enemy.dy, Ball.x, Ball.y, Ball.dx, Ball.dy]
    Normalized deterministically to [0, 1]:
      - Paddle Y (center): (y - h/2) / (SCREEN_H - h)
      - Ball X/Y: x / (SCREEN_W - 1), y / (SCREEN_H - 1)
      - Velocities: symmetric tanh mapping -> [0,1]
    """

    # Index objects by category for easy lookup
    obj_map = _index_objects_by_category(objects)

    # Retrieve player state and normalize
    player_obj = obj_map["Player"]
    player_y = player_obj.y
    player_dy = player_obj.dy
    player_h = _paddle_height(player_obj)
    player_y_n = _normalize_paddle_center_y(player_y, player_h)
    player_dy_n = _normalize_velocity(player_dy, PADDLE_DY_SCALE)

    # Retrieve enemy state and normalize
    enemy_obj = obj_map.get("Enemy", None)
    enemy_y = enemy_obj.y if enemy_obj else 0
    enemy_dy = enemy_obj.dy if enemy_obj else 0
    if enemy_obj is not None:
        enemy_h = _paddle_height(enemy_obj)
        enemy_y_n = _normalize_paddle_center_y(enemy_y, enemy_h)
    else:
        enemy_y_n = 0.0
    enemy_dy_n = _normalize_velocity(enemy_dy, PADDLE_DY_SCALE)

    # Retrieve ball state and normalize
    ball_obj = obj_map.get("Ball", None)
    ball_x = ball_obj.x if ball_obj else 0
    ball_y = ball_obj.y if ball_obj else 0
    ball_dx = ball_obj.dx if ball_obj else 0
    ball_dy = ball_obj.dy if ball_obj else 0
    # Normalize ball positions over pixel index ranges [0..SCREEN_W-1], [0..SCREEN_H-1]
    ball_x_n = (ball_x / (SCREEN_W - 1.0)) if ball_obj else 0.0
    ball_y_n = (ball_y / (SCREEN_H - 1.0)) if ball_obj else 0.0
    ball_dx_n = _normalize_velocity(ball_dx, BALL_D_SCALE)
    ball_dy_n = _normalize_velocity(ball_dy, BALL_D_SCALE)

    # Create state vector and return it
    obs = np.asarray([
        player_y_n, 
        player_dy_n, 
        enemy_y_n,
        enemy_dy_n, 
        ball_x_n, ball_y_n, 
        ball_dx_n, ball_dy_n
    ], dtype=np.float32)
    return obs

# ---- Gymnasium Observation Wrapper ----
class PongV5_FeatureExtractor(gym.ObservationWrapper):
    """
    Replaces obs with an 8-dim normalized vector built from OCAtari objects.
    """
    def __init__(self, env):
        super().__init__(env)
        # TODO: if not normalized then return correct max values for each feature
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def observation(self, observation):
        # Convert objects to observation vector
        obs = _obs_from_objects(self.env.objects)
        
        # Assert that obs is within [0, 1] and finite and not nan
        assert np.all(obs >= 0.0), f"Negative values found: {obs}"
        assert np.all(obs <= 1.0), f"Values above 1 found: {obs}"
        assert np.all(np.isfinite(obs)), f"Non-finite values found: {obs}"

        # Return observation
        return obs
