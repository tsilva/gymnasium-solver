import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Iterable, Optional, Tuple

from gym_wrappers.ocatari_helpers import center, normalize_velocity

SCREEN_W: float = 160.0
SCREEN_H: float = 210.0
PLAYFIELD_X_MIN: float = 0.0
PLAYFIELD_X_MAX: float = SCREEN_W
PLAYFIELD_Y_MIN: float = 32.0
PLAYFIELD_Y_MAX: float = 200.0
PADDLE_MARGIN_X: float = 2.0
BALL_MARGIN: float = 1.0
PADDLE_DX_SCALE: float = 6.0
BALL_D_SCALE: float = 10.0
MAX_BLOCKS: int = 108


def _categorise_objects(objects: Iterable):
    player = None
    ball = None
    blocks = []
    for obj in objects:
        if not obj or getattr(obj, "hud", False):
            continue
        category = getattr(obj, "category", None)
        if category == "Player" and player is None:
            player = obj
        elif category == "Ball" and ball is None:
            ball = obj
        elif category == "Block":
            blocks.append(obj)
    return player, ball, blocks


def _normalize_linear(value: float, lo: float, hi: float) -> float:
    assert hi > lo, f"Invalid range [{lo}, {hi}]"
    zero_one = (value - lo) / (hi - lo)
    return 2.0 * zero_one - 1.0


def _obs_from_objects(
    objects,
    last_ball_x_n: Optional[float] = None,
    last_ball_y_n: Optional[float] = None,
    last_ball_dx_n: Optional[float] = None,
    last_ball_dy_n: Optional[float] = None,
):
    player, ball, blocks = _categorise_objects(objects)

    if player is not None:
        paddle_cx, paddle_cy = center(player)
        paddle_w = float(getattr(player, "w", 16.0))
        paddle_dx = float(getattr(player, "dx", 0.0))
    else:
        paddle_cx, paddle_cy = SCREEN_W / 2.0, PLAYFIELD_Y_MAX
        paddle_w = 16.0
        paddle_dx = 0.0

    paddle_x_n = _normalize_linear(
        paddle_cx,
        PLAYFIELD_X_MIN + 0.5 * paddle_w - PADDLE_MARGIN_X,
        PLAYFIELD_X_MAX - 0.5 * paddle_w + PADDLE_MARGIN_X,
    )
    paddle_y_n = _normalize_linear(paddle_cy, PLAYFIELD_Y_MIN, PLAYFIELD_Y_MAX)
    paddle_dx_n = normalize_velocity(paddle_dx, PADDLE_DX_SCALE)

    ball_visible = bool(ball)
    if ball_visible:
        ball_cx, ball_cy = center(ball)
        ball_dx_px = float(getattr(ball, "dx", 0.0))
        ball_dy_px = float(getattr(ball, "dy", 0.0))
        ball_x_n = _normalize_linear(
            ball_cx,
            PLAYFIELD_X_MIN + BALL_MARGIN,
            PLAYFIELD_X_MAX - BALL_MARGIN,
        )
        ball_y_n = _normalize_linear(
            ball_cy,
            PLAYFIELD_Y_MIN + BALL_MARGIN,
            PLAYFIELD_Y_MAX - BALL_MARGIN,
        )
    else:
        ball_cx = ball_cy = None
        ball_dx_px = ball_dy_px = 0.0
        ball_x_n = float(last_ball_x_n) if last_ball_x_n is not None else 0.0
        ball_y_n = float(last_ball_y_n) if last_ball_y_n is not None else -1.0

    if ball_visible and last_ball_x_n is not None and last_ball_y_n is not None:
        ball_dx_n = float(ball_x_n) - float(last_ball_x_n)
        ball_dy_n = float(ball_y_n) - float(last_ball_y_n)
    elif ball_visible:
        ball_dx_n = normalize_velocity(ball_dx_px, BALL_D_SCALE)
        ball_dy_n = normalize_velocity(ball_dy_px, BALL_D_SCALE)
    else:
        ball_dx_n = float(last_ball_dx_n) if last_ball_dx_n is not None else 0.0
        ball_dy_n = float(last_ball_dy_n) if last_ball_dy_n is not None else 0.0

    ball_visible_n = 1.0 if ball_visible else -1.0

    rel_ball_x_n = 0.5 * (ball_x_n - paddle_x_n)
    rel_ball_y_n = 0.5 * (ball_y_n - paddle_y_n)
    rel_ball_x_n = float(np.clip(rel_ball_x_n, -1.0, 1.0))
    rel_ball_y_n = float(np.clip(rel_ball_y_n, -1.0, 1.0))

    block_positions = {(int(getattr(b, "x", 0)), int(getattr(b, "y", 0))) for b in blocks}
    blocks_remaining = min(len(block_positions), MAX_BLOCKS)
    blocks_remaining_n = 2.0 * (blocks_remaining / MAX_BLOCKS) - 1.0

    obs = np.asarray([
        paddle_x_n,
        paddle_dx_n,
        ball_x_n,
        ball_y_n,
        ball_dx_n,
        ball_dy_n,
        ball_visible_n,
        rel_ball_x_n,
        rel_ball_y_n,
        blocks_remaining_n,
    ], dtype=np.float32)

    assert np.all(np.isfinite(obs)), f"Breakout feature extractor produced non-finite obs: {obs}"
    return obs


class BreakoutV5_FeatureExtractor(gym.ObservationWrapper):
    """Replace Breakout observations with structured OCAtari features."""

    def __init__(self, env, clip: bool = True):
        super().__init__(env)
        self.clip = bool(clip)
        self._last_ball_x_n: Optional[float] = None
        self._last_ball_y_n: Optional[float] = None
        self._last_ball_dx_n: Optional[float] = None
        self._last_ball_dy_n: Optional[float] = None
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

    def reset(self, **kwargs):
        self._last_ball_x_n = None
        self._last_ball_y_n = None
        self._last_ball_dx_n = None
        self._last_ball_dy_n = None
        return super().reset(**kwargs)

    def observation(self, observation):
        obs = _obs_from_objects(
            self.env.objects,
            last_ball_x_n=self._last_ball_x_n,
            last_ball_y_n=self._last_ball_y_n,
            last_ball_dx_n=self._last_ball_dx_n,
            last_ball_dy_n=self._last_ball_dy_n,
        )
        if obs[6] > 0.0:
            self._last_ball_x_n = float(obs[2])
            self._last_ball_y_n = float(obs[3])
            self._last_ball_dx_n = float(obs[4])
            self._last_ball_dy_n = float(obs[5])
        if self.clip:
            obs = np.clip(obs, -1.0, 1.0, out=obs)
        return obs
