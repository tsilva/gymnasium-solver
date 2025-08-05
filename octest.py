import numpy as np
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # disables SDL sound

MIN_VEC = np.array([34, -23, 0, -8, 0, 0, -12, -8], dtype=float)
MAX_VEC = np.array([190, 24, 191, 72, 156, 191, 12, 9], dtype=float)

# Fixed order for Pong: Player paddle, Enemy paddle, Ball
# Each contributes [x, y, w, h, dx, dy] → total length = 18
PONG_ORDER = ("Player", "Enemy", "Ball")

def index_objects(objects, include_hud=False):
    """
    Index OCAtari objects by their category, excluding HUD if specified.
    Returns a dictionary mapping category names to objects.
    """
    obj_map = {}
    for o in objects:
        if not include_hud and getattr(o, "hud", False):
            continue
        if o.category not in obj_map:
            obj_map[o.category] = o
    return obj_map

def pong_state_vector(objects, include_hud=False):
    """
    Build a 1D float32 vector from OCAtari objects for Pong.
    Order per object: [x, y, w, h, dx, dy] for Player, Enemy, Ball.
    Missing objects are zero-padded.
    """
    obj_map = index_objects(objects, include_hud)

    feats = []
    for name in PONG_ORDER:
        obj = obj_map.get(name, None)
        if name == "Ball":
            if obj is None:
                feats.extend([0, 0, 0, 0])
            else:
                feats.extend([obj.x, obj.y, obj.dx, obj.dy])
        else:
            if obj is None:
                feats.extend([0, 0])
            else:
                feats.extend([obj.y, obj.dy])
    features = np.asarray(feats, dtype=np.float32)
    features = (features - MIN_VEC) / (MAX_VEC - MIN_VEC)  # Normalize
    return features

# --- usage in your snippet ---
from ocatari.core import OCAtari
import random

env = OCAtari("ALE/Pong-v5", mode="ram", hud=False, render_mode="human")
obs, info = env.reset()

vecs = []
action = random.randint(0, env.nb_actions - 1)
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    vec = pong_state_vector(env.objects, include_hud=False)  # shape (18,)
    vecs.append(vec)
    print(vec)
    if terminated or truncated:
        break

print(np.min(vecs, axis=0))
print(np.max(vecs, axis=0))

min_vec = [ 34, -23,   0,  -8,   0,   0, -12,  -8]
max_vec = [190,  24, 191,  72, 156, 191,  12,   9]