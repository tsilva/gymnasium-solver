import gymnasium as gym
import numpy as np
from gymnasium import spaces

# TODO: CLEANUP this file

# Deterministic normalization constants to keep features in [0,1]
# Screen dimensions for Atari Pong (overshoot-safe denominators)
SCREEN_W: float = 160.0
SCREEN_H: float = 210.0

# Symmetric velocity scales. Tanh-based mapping guarantees [0,1].
# Choose conservative scales to avoid saturation for typical speeds.
PADDLE_DY_SCALE: float = 24.0
BALL_D_SCALE: float = 12.0

# ---- helpers from your snippet ----
def _index_objects_by_category(objects, include_hud: bool = False):
    objects_map = {}
    for object in objects:
        if not include_hud and getattr(object, "hud", False): continue
        if object.category not in objects_map: objects_map[object.category] = object
    return objects_map

def _sym_to_unit(value: float, scale: float) -> float:
    """Map symmetric value in R to [0,1] using tanh with scale.

    0 maps to 0.5, extremes saturate to 0/1.
    """
    return 0.5 * (np.tanh(float(value) / float(scale)) + 1.0)


def pong_state_vector(objects, include_hud: bool = False):
    """
    Vector order (length=8): [Player.y, Player.dy, Enemy.y, Enemy.dy, Ball.x, Ball.y, Ball.dx, Ball.dy]
    Normalized deterministically to [0, 1]:
      - Positions: divide by screen size (x/SCREEN_W, y/SCREEN_H)
      - Velocities: symmetric tanh mapping -> [0,1]
    """

    # Index objects by category for easy lookup
    obj_map = _index_objects_by_category(objects, include_hud=include_hud)

    # Retrieve player state and normalize
    player_obj = obj_map["Player"]
    player_y = player_obj.y
    player_dy = player_obj.dy
    player_y_n = player_y / SCREEN_H
    player_dy_n = _sym_to_unit(player_dy, PADDLE_DY_SCALE)

    # Retrieve enemy state and normalize
    enemy_obj = obj_map["Enemy"]
    enemy_y = enemy_obj.y
    enemy_dy = enemy_obj.dy
    enemy_y_n = enemy_y / SCREEN_H
    enemy_dy_n = _sym_to_unit(enemy_dy, PADDLE_DY_SCALE)

    # Retrieve ball state and normalize
    ball_obj = obj_map["Ball"]
    ball_x = ball_obj.x
    ball_y = ball_obj.y
    ball_dx = ball_obj.dx
    ball_dy = ball_obj.dy
    ball_x_n = ball_x / SCREEN_W
    ball_y_n = ball_y / SCREEN_H
    ball_dx_n = _sym_to_unit(ball_dx, BALL_D_SCALE)
    ball_dy_n = _sym_to_unit(ball_dy, BALL_D_SCALE)

    # Create state vector and return it
    state_vector = np.asarray([
        player_y_n, 
        player_dy_n, 
        enemy_y_n,
        enemy_dy_n, 
        ball_x_n, ball_y_n, 
        ball_dx_n, ball_dy_n
    ], dtype=np.float32)
    return state_vector

# ---- Gymnasium Observation Wrapper ----
class PongV5_FeatureExtractor(gym.ObservationWrapper):
    """
    Replaces obs with an 8-dim normalized vector built from OCAtari objects.
    """
    def __init__(self, env, include_hud: bool = False, clip: bool = True):
        super().__init__(env)
        self.include_hud = include_hud
        self.clip = clip
        # TODO: if not normalized then return correct max values for each feature
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def observation(self, observation):
        vec = pong_state_vector(self.env.objects, include_hud=self.include_hud)
        
        # TODO: review clipping, consider using assertion instead
        #assert np.all(vec > 0.0) and np.all(vec < 1.0)

        if self.clip:
            # Check before clipping
            if np.any(vec < 0.0) or np.any(vec > 1.0):
                print(f"[CLIP] Raw vector out of bounds: {vec}")
            vec = np.clip(vec, 0.0, 1.0)

        return vec
