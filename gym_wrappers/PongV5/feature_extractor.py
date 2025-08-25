import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Deterministic normalization constants to keep features in [0,1]
# Screen dimensions for Atari Pong (overshoot-safe denominators)
SCREEN_W: float = 160.0
SCREEN_H: float = 210.0

# Symmetric velocity scales. Tanh-based mapping guarantees [0,1].
# Choose conservative scales to avoid saturation for typical speeds.
PADDLE_DY_SCALE: float = 24.0
BALL_D_SCALE: float = 12.0

# ---- helpers from your snippet ----
def index_objects(objects, include_hud: bool = False):
    obj_map = {}
    for o in objects:
        if not include_hud and getattr(o, "hud", False):
            continue
        if o.category not in obj_map:
            obj_map[o.category] = o
    return obj_map

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
    obj_map = index_objects(objects, include_hud)

    # Player
    p = obj_map.get("Player", None)
    p_y = float(getattr(p, "y", 0.0))
    p_dy = float(getattr(p, "dy", 0.0))
    p_y_n = p_y / SCREEN_H
    p_dy_n = _sym_to_unit(p_dy, PADDLE_DY_SCALE)

    # Enemy
    e = obj_map.get("Enemy", None)
    e_y = float(getattr(e, "y", 0.0))
    e_dy = float(getattr(e, "dy", 0.0))
    e_y_n = e_y / SCREEN_H
    e_dy_n = _sym_to_unit(e_dy, PADDLE_DY_SCALE)

    # Ball
    b = obj_map.get("Ball", None)
    b_x = float(getattr(b, "x", 0.0))
    b_y = float(getattr(b, "y", 0.0))
    b_dx = float(getattr(b, "dx", 0.0))
    b_dy = float(getattr(b, "dy", 0.0))
    b_x_n = b_x / SCREEN_W
    b_y_n = b_y / SCREEN_H
    b_dx_n = _sym_to_unit(b_dx, BALL_D_SCALE)
    b_dy_n = _sym_to_unit(b_dy, BALL_D_SCALE)

    features = np.asarray([p_y_n, p_dy_n, e_y_n, e_dy_n, b_x_n, b_y_n, b_dx_n, b_dy_n], dtype=np.float32)
    return features

# ---- Gymnasium Observation Wrapper ----
class PongV5_FeatureExtractor(gym.ObservationWrapper):
    """
    Replaces obs with an 8-dim normalized vector built from OCAtari objects.
    """
    def __init__(self, env, include_hud: bool = False, clip: bool = True):
        super().__init__(env)
        self.include_hud = include_hud
        self.clip = clip
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def observation(self, observation):
        vec = pong_state_vector(self.env.objects, include_hud=self.include_hud)

        if self.clip:
            # Check before clipping
            if np.any(vec < 0.0) or np.any(vec > 1.0):
                print(f"[CLIP] Raw vector out of bounds: {vec}")
            vec = np.clip(vec, 0.0, 1.0)

        return vec
