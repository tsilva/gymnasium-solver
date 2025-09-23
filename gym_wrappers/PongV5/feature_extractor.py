import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Screen dimensions for Atari Pong (for normalizing positions)
SCREEN_W: float = 160.0
SCREEN_H: float = 210.0

# Effective playable vertical range for Pong (excludes HUD/borders).
# Common Atari preprocessing crops ~34px from top and ~16px from bottom.
# This yields a 160px-tall playfield: [34, 194) in screen coordinates.
PLAYFIELD_Y_MIN: float = 34.0
PLAYFIELD_Y_MAX: float = 194.0
PLAYFIELD_H: float = PLAYFIELD_Y_MAX - PLAYFIELD_Y_MIN

# Sanity checks to avoid silent miscalibration
assert 0.0 <= PLAYFIELD_Y_MIN < PLAYFIELD_Y_MAX <= SCREEN_H, (
    f"Invalid Pong playfield bounds: [{PLAYFIELD_Y_MIN}, {PLAYFIELD_Y_MAX}) for SCREEN_H={SCREEN_H}"
)

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
    """Map symmetric value in R to [-1, 1] using tanh with scale.

    0 maps to 0.0, extremes saturate to -1/1.
    """
    return float(np.tanh(float(value) / float(scale)))

def _center_y(obj) -> float:
    """Return robust center-Y for an OCAtari object.

    Prefer explicit center if present; otherwise derive from top-left y and height.
    """
    if obj is None:
        return 0.0
    # OCAtari objects often provide `center` (x, y); fall back to y + h/2.
    if hasattr(obj, "center") and obj.center is not None:
        return float(obj.center[1])
    h = float(getattr(obj, "h", 0.0))
    y = float(getattr(obj, "y", 0.0))
    return y + 0.5 * h

def _normalize_paddle_center_y(center_y: float, paddle_h: float,
                               min_y: float, max_y: float, margin_y: float) -> float:
    """Normalize paddle center-Y to [-1, 1] over Pong's playable range.

    Uses playable center range: [min_y + h/2 - margin, max_y - h/2 + margin].
    A small margin (in pixels) provides tolerance at borders.
    """
    lo = min_y + 0.5 * paddle_h - margin_y
    hi = max_y - 0.5 * paddle_h + margin_y
    denom = hi - lo
    assert denom > 0.0, (
        f"Invalid paddle center range: [{lo}, {hi}] for h={paddle_h}, min_y={min_y}, max_y={max_y}"
    )
    zero_one = (center_y - lo) / denom
    return 2.0 * zero_one - 1.0


def _obs_from_objects(objects,
                      min_y: float = PLAYFIELD_Y_MIN,
                      max_y: float = PLAYFIELD_Y_MAX,
                      margin_y: float = 2.0):
    """
    Vector order (length=9):
      [Player.y, Player.dy, Enemy.y, Enemy.dy, Ball.x, Ball.y, Ball.dx, Ball.dy, Ball.visible]

    Normalized deterministically to [-1, 1]:
      - Paddle Y (center): over [PLAYFIELD_Y_MIN + h/2, PLAYFIELD_Y_MAX - h/2]
        mapped linearly to [-1,1]
      - Ball X/Y: X over [0..SCREEN_W-1], Y over [PLAYFIELD_Y_MIN..PLAYFIELD_Y_MAX-1]
        mapped linearly to [-1,1]
      - Velocities: symmetric tanh mapping -> [-1,1]
      - Ball.visible flag: +1 if ball is visible/present, -1 if absent
    """

    # Index objects by category for easy lookup
    obj_map = _index_objects_by_category(objects)

    # Retrieve player state and normalize
    player_obj = obj_map["Player"]
    player_h = float(player_obj.h)
    player_center_y = _center_y(player_obj)
    player_dy = player_obj.dy
    player_y_n = _normalize_paddle_center_y(player_center_y, player_h, min_y, max_y, margin_y)
    player_dy_n = _normalize_velocity(player_dy, PADDLE_DY_SCALE)

    # Retrieve enemy state and normalize
    enemy_obj = obj_map.get("Enemy", None)
    enemy_dy = enemy_obj.dy if enemy_obj else 0
    if enemy_obj is not None:
        enemy_h = float(enemy_obj.h)
        enemy_center_y = _center_y(enemy_obj)
        enemy_y_n = _normalize_paddle_center_y(enemy_center_y, enemy_h, min_y, max_y, margin_y)
    else:
        enemy_y_n = 0.0
    enemy_dy_n = _normalize_velocity(enemy_dy, PADDLE_DY_SCALE)

    # Retrieve ball state and normalize
    ball_obj = obj_map.get("Ball", None)
    ball_x = ball_obj.x if ball_obj else 0
    ball_center_y = _center_y(ball_obj) if ball_obj else 0.0
    ball_dx = ball_obj.dx if ball_obj else 0
    ball_dy = ball_obj.dy if ball_obj else 0
    ball_visible = (ball_obj is not None) and bool(getattr(ball_obj, "visible", True))
    # Normalize ball positions: X over [0..SCREEN_W-1], Y over Pong playfield using center-Y
    if ball_obj:
        bx01 = ball_x / (SCREEN_W - 1.0)
        h = float(ball_obj.h)
        lo = min_y + 0.5 * h - margin_y
        hi = max_y - 0.5 * h + margin_y
        denom = hi - lo
        assert denom > 0.0, (
            f"Invalid ball center range: [{lo}, {hi}] for h={h}, min_y={min_y}, max_y={max_y}"
        )
        by01 = (ball_center_y - lo) / denom
        ball_x_n = 2.0 * bx01 - 1.0
        ball_y_n = 2.0 * by01 - 1.0
    else:
        ball_x_n = 0.0
        ball_y_n = 0.0
    ball_dx_n = _normalize_velocity(ball_dx, BALL_D_SCALE)
    ball_dy_n = _normalize_velocity(ball_dy, BALL_D_SCALE)
    ball_visible_n = 1.0 if ball_visible else -1.0

    # Create state vector and return it
    obs = np.asarray([
        player_y_n, 
        player_dy_n, 
        enemy_y_n,
        enemy_dy_n, 
        ball_x_n, ball_y_n, 
        ball_dx_n, ball_dy_n,
        ball_visible_n,
    ], dtype=np.float32)
    return obs

# ---- Gymnasium Observation Wrapper ----
class PongV5_FeatureExtractor(gym.ObservationWrapper):
    """
    Replaces obs with a 9-dim normalized vector built from OCAtari objects.
    """
    def __init__(self, env,
                 clip: bool = True,
                 playfield_y_min: float = PLAYFIELD_Y_MIN,
                 playfield_y_max: float = PLAYFIELD_Y_MAX,
                 margin_y: float = 2.0):
        super().__init__(env)
        self.clip = bool(clip)
        self.min_y = float(playfield_y_min)
        self.max_y = float(playfield_y_max)
        self.margin_y = float(margin_y)
        # Observation space remains [-1,1] as features are designed to be scaled there.
        # When clip=False, features may briefly exceed this range, but most algorithms
        # are tolerant as long as the Box advertises the intended scale.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    def observation(self, observation):
        # Convert objects to observation vector using configured bounds
        obs = _obs_from_objects(self.env.objects, self.min_y, self.max_y, self.margin_y)
        # Optionally clip to the target range to avoid rare excursions at borders
        if self.clip:
            obs = np.clip(obs, -1.0, 1.0, out=obs)
        # Always ensure finiteness
        assert np.all(np.isfinite(obs)), f"Non-finite values found: {obs}"
        return obs
