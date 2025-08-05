import numpy as np

# Fixed order for Pong: Player paddle, Enemy paddle, Ball
# Each contributes [x, y, w, h, dx, dy] → total length = 18
PONG_ORDER = ("Player", "Enemy", "Ball")

def pong_state_vector(objects, include_hud=False):
    """
    Build a 1D float32 vector from OCAtari objects for Pong.
    Order per object: [x, y, w, h, dx, dy] for Player, Enemy, Ball.
    Missing objects are zero-padded.
    """
    feats = []
    for name in PONG_ORDER:
        obj = next((o for o in objects if o.category == name and (include_hud or not getattr(o, "hud", False))), None)
        if name == "Ball":  
            if obj is None: feats.extend([0, 0, 0, 0])
            else: feats.extend([obj.x, obj.u, obj.dx, obj.dy])
        else:
            if obj is None: feats.extend([0, 0]) 
            else: feats.extend([obj.y, obj.dy])
    return np.asarray(feats, dtype=np.float32)

# --- usage in your snippet ---
from ocatari.core import OCAtari
import random

env = OCAtari("ALE/Pong-v5", mode="ram", hud=True, render_mode="rgb_array")
obs, info = env.reset()

action = random.randint(0, env.nb_actions - 1)
obs, reward, terminated, truncated, info = env.step(action)

vec = pong_state_vector(env.objects, include_hud=False)  # shape (18,)
print(vec)