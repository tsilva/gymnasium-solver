import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Normalization constants for Pong state vector, manually 
# computed through random play of a single episode
MIN_VEC = np.array([34, -23,   0,  -8,   0,   0, -12,  -8], dtype=float)
MAX_VEC = np.array([190,  24, 191,  72, 156, 191,  12,   9], dtype=float)

# ---- helpers from your snippet ----
def index_objects(objects, include_hud: bool = False):
    obj_map = {}
    for o in objects:
        if not include_hud and getattr(o, "hud", False):
            continue
        if o.category not in obj_map:
            obj_map[o.category] = o
    return obj_map

def pong_state_vector(objects, include_hud: bool = False):
    """
    Vector order (length=8): [Player.y, Player.dy, Enemy.y, Enemy.dy, Ball.x, Ball.y, Ball.dx, Ball.dy]
    Normalized to [0, 1] with MIN_VEC/MAX_VEC.
    """
    obj_map = index_objects(objects, include_hud)

    feats = []
    # Player
    p = obj_map.get("Player", None)
    feats.extend([0, 0] if p is None else [p.y, p.dy])
    # Enemy
    e = obj_map.get("Enemy", None)
    feats.extend([0, 0] if e is None else [e.y, e.dy])
    # Ball
    b = obj_map.get("Ball", None)
    feats.extend([0, 0, 0, 0] if b is None else [b.x, b.y, b.dx, b.dy])

    features = np.asarray(feats, dtype=np.float32)
    features = (features - MIN_VEC) / (MAX_VEC - MIN_VEC)  # normalize
    return features.astype(np.float32)

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
            vec = np.clip(vec, 0.0, 1.0)
        return vec
