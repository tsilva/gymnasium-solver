import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Sequence

from gym_wrappers.ocatari_helpers import center_x, center_y, normalize_linear, normalize_position, normalize_velocity, index_objects_by_category


def _find_env_with_objects(env):
    """Drill through wrapper stack to find environment with 'objects' attribute."""
    current = env
    while current is not None:
        if hasattr(current, 'objects'):
            return current
        current = getattr(current, 'env', None)
    return None


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
# Treat very small paddle velocity as stationary to avoid spurious action flips.
PADDLE_STILL_EPS: float = 1e-3

# Max X/Y pixels per frame that ball can move (for normalizing velocity)
BALL_D_SCALE: float = 12.0

# (removed _normalize_paddle_center_y; now using normalize_position from ocatari_helpers)


def _obs_from_objects(
    objects,
    min_y: float = PLAYFIELD_Y_MIN,
    max_y: float = PLAYFIELD_Y_MAX,
    margin_y: float = 2.0,
    # Optional last-seen, normalized ball state for invisible frames
    last_ball_x_n: Optional[float] = None,
    last_ball_y_n: Optional[float] = None,
    last_ball_dx_n: Optional[float] = None,
    last_ball_dy_n: Optional[float] = None,
):
    """
    Vector order (length=9):
      [Player.y, Player.dy, Enemy.y, Enemy.dy, Ball.x, Ball.y, Ball.dx, Ball.dy, Ball.visible]

    Normalized deterministically to [-1, 1]:
      - Paddle Y (center): over [PLAYFIELD_Y_MIN + h/2, PLAYFIELD_Y_MAX - h/2]
        mapped linearly to [-1,1]
      - Ball X/Y: center-based. X over [w/2..SCREEN_W - w/2] and
        Y over [PLAYFIELD_Y_MIN + h/2 .. PLAYFIELD_Y_MAX - h/2], each with a
        small margin, mapped linearly to [-1,1]
      - Velocities (dx, dy): expressed in the same normalized coordinate
        system as positions so that, when consecutive frames are available,
        x_t+1 ≈ x_t + dx and y_t+1 ≈ y_t + dy.
      - Ball.visible flag: +1 if ball is visible/present, -1 if absent
    """

    # Index objects by category for easy lookup
    obj_map = index_objects_by_category(objects)

    # Retrieve player state and normalize
    player_obj = obj_map["Player"]
    player_h = float(player_obj.h)
    player_center_y_val = center_y(player_obj)
    player_dy = player_obj.dy
    player_y_n = normalize_position(player_center_y_val, min_y, max_y, player_h, margin_y)
    player_dy_n = normalize_velocity(player_dy, PADDLE_DY_SCALE)

    # Retrieve enemy state and normalize
    enemy_obj = obj_map.get("Enemy", None)
    enemy_dy = enemy_obj.dy if enemy_obj else 0
    if enemy_obj is not None:
        enemy_h = float(enemy_obj.h)
        enemy_center_y_val = center_y(enemy_obj)
        enemy_y_n = normalize_position(enemy_center_y_val, min_y, max_y, enemy_h, margin_y)
    else:
        enemy_y_n = 0.0
    enemy_dy_n = normalize_velocity(enemy_dy, PADDLE_DY_SCALE)

    # Retrieve ball state and normalize
    ball_obj = obj_map.get("Ball", None)
    ball_center_x_val = center_x(ball_obj) if ball_obj else 0.0
    ball_center_y_val = center_y(ball_obj) if ball_obj else 0.0
    ball_dx = ball_obj.dx if ball_obj else 0
    ball_dy = ball_obj.dy if ball_obj else 0
    ball_visible = (ball_obj is not None) and bool(getattr(ball_obj, "visible", True))

    # Normalize ball positions: use center-X and center-Y with width/height-aware bounds
    if ball_visible:
        # X normalization across the full screen width, adjusted for ball width
        w = float(getattr(ball_obj, "w", 0.0))
        ball_x_n = normalize_position(ball_center_x_val, 0.0, SCREEN_W, w, margin_y)

        # Y normalization within playable field, adjusted for ball height
        h = float(getattr(ball_obj, "h", 0.0))
        ball_y_n = normalize_position(ball_center_y_val, min_y, max_y, h, margin_y)
    else:
        # Use last-seen normalized positions if provided; fallback to 0.0
        ball_x_n = float(last_ball_x_n) if last_ball_x_n is not None else 0.0
        ball_y_n = float(last_ball_y_n) if last_ball_y_n is not None else 0.0

    if ball_visible:
        # Compute velocities in normalized coordinates so that
        # delta(position_normalized) == velocity_normalized.
        if last_ball_x_n is not None and last_ball_y_n is not None:
            ball_dx_n = float(ball_x_n) - float(last_ball_x_n)
            ball_dy_n = float(ball_y_n) - float(last_ball_y_n)
        else:
            # First visible frame: approximate using pixel velocities scaled to normalized units
            # using the same denominators used for positions above.
            # Guard denominators: if not previously defined (invisible path), estimate using
            # nominal ball size of 2px.
            if ball_obj:
                w = float(getattr(ball_obj, "w", 2.0))
                h = float(getattr(ball_obj, "h", 2.0))
            else:
                w = 2.0
                h = 2.0
            # Compute range using same logic as normalize_position
            x_lo = 0.0 + 0.5 * w - margin_y
            x_hi = SCREEN_W - 0.5 * w + margin_y
            y_lo = min_y + 0.5 * h - margin_y
            y_hi = max_y - 0.5 * h + margin_y
            x_denom = x_hi - x_lo
            y_denom = y_hi - y_lo
            assert x_denom > 0.0 and y_denom > 0.0, (
                f"Invalid ball velocity denominator: x_denom={x_denom}, y_denom={y_denom}"
            )
            ball_dx_n = 2.0 * float(ball_dx) / float(x_denom)
            ball_dy_n = 2.0 * float(ball_dy) / float(y_denom)
    else:
        # Use last-seen normalized velocities if provided; fallback to 0.0
        ball_dx_n = float(last_ball_dx_n) if last_ball_dx_n is not None else 0.0
        ball_dy_n = float(last_ball_dy_n) if last_ball_dy_n is not None else 0.0
        
    ball_visible_n = 1.0 if ball_visible else -1.0

    # Invariant: if we have a previous position and current ball is visible,
    # the normalized position should advance by the normalized velocity.
    if ball_visible and last_ball_x_n is not None and last_ball_y_n is not None:
        # Allow tiny numerical slack due to float ops
        eps = 1e-6
        assert abs((float(last_ball_x_n) + float(ball_dx_n)) - float(ball_x_n)) <= max(eps, 1e-3)
        assert abs((float(last_ball_y_n) + float(ball_dy_n)) - float(ball_y_n)) <= max(eps, 1e-3)

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
    Replaces obs with a normalized vector built from OCAtari objects plus last-action one-hot.
    """
    def __init__(self, env,
                 clip: bool = True,
                 playfield_y_min: float = PLAYFIELD_Y_MIN,
                 playfield_y_max: float = PLAYFIELD_Y_MAX,
                 margin_y: float = 2.0,
                 action_ids: Optional[Sequence[int]] = None):
        super().__init__(env)

        self.clip = bool(clip)
        self.min_y = float(playfield_y_min)
        self.max_y = float(playfield_y_max)
        self.margin_y = float(margin_y)

        # Determine which discrete actions to encode in the one-hot tail of the observation.
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError(
                f"PongV5_FeatureExtractor requires a Discrete action space, got {type(env.action_space)}"
            )

        if action_ids is None:
            self._action_ids: tuple[int, ...] = tuple(range(int(env.action_space.n)))
        else:
            if len(action_ids) == 0:
                raise ValueError("action_ids must contain at least one action index")
            self._action_ids = tuple(int(a) for a in action_ids)

        if len(set(self._action_ids)) != len(self._action_ids):
            raise ValueError(f"action_ids must be unique, got {self._action_ids}")

        # Map base-environment action indices to positions in the one-hot vector.
        self._action_index_map: dict[int, int] = {
            action: idx for idx, action in enumerate(self._action_ids)
        }
        self._last_action_one_hot = np.zeros(len(self._action_ids), dtype=np.float32)
        # Last-seen normalized ball kinematics (persist across steps; reset on env.reset)
        self._last_ball_x_n: Optional[float] = None
        self._last_ball_y_n: Optional[float] = None
        self._last_ball_dx_n: Optional[float] = None
        self._last_ball_dy_n: Optional[float] = None
        # Observation space remains [-1,1] as features are designed to be scaled there.
        # When clip=False, features may briefly exceed this range, but most algorithms
        # are tolerant as long as the Box advertises the intended scale.
        base_low = np.full(9, -1.0, dtype=np.float32)
        base_high = np.full(9, 1.0, dtype=np.float32)
        action_low = np.zeros(len(self._action_ids), dtype=np.float32)
        action_high = np.ones(len(self._action_ids), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, action_low]),
            high=np.concatenate([base_high, action_high]),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        # Clear last-seen ball state between episodes
        self._last_ball_x_n = None
        self._last_ball_y_n = None
        self._last_ball_dx_n = None
        self._last_ball_dy_n = None
        self._last_action_one_hot.fill(0.0)
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        executed_action = self._infer_executed_action(
            fallback=int(np.asarray(action).item())
        )
        self._record_last_action(executed_action)
        obs = self.observation(obs)
        return obs, reward, terminated, truncated, info

    def _record_last_action(self, action) -> None:
        try:
            vector_index = self._action_index_map[int(action)]
        except KeyError:
            return
        self._last_action_one_hot.fill(0.0)
        self._last_action_one_hot[vector_index] = 1.0

    def _infer_executed_action(self, fallback: int) -> int:
        """Infer the action ALE executed after stickiness by observing paddle velocity."""
        ocatari_env = _find_env_with_objects(self.env)
        if ocatari_env is None:
            return fallback
        objects = ocatari_env.objects

        try:
            obj_map = index_objects_by_category(objects)
        except Exception:
            return fallback

        player_obj = obj_map.get("Player")
        if player_obj is None:
            return fallback

        dy = float(getattr(player_obj, "dy", 0.0))

        # When the paddle is stationary, NOOP (action 0) must have executed.
        # This is true whether we requested NOOP (correct) or requested movement
        # (sticky action caused NOOP to execute instead).
        if abs(dy) <= PADDLE_STILL_EPS:
            return 0  # NOOP

        # Paddle is moving: infer direction from velocity
        moving_down = dy > 0.0
        return 3 if moving_down else 2  # 3=down action, 2=up action

    def observation(self, observation):
        # Convert objects to observation vector using configured bounds.
        # When the ball is invisible, use last-seen kinematics to avoid collisions with valid zeros.
        ocatari_env = _find_env_with_objects(self.env)
        assert ocatari_env is not None, "Could not find environment with 'objects' attribute in wrapper stack"
        base_obs = _obs_from_objects(
            ocatari_env.objects,
            self.min_y,
            self.max_y,
            self.margin_y,
            last_ball_x_n=self._last_ball_x_n,
            last_ball_y_n=self._last_ball_y_n,
            last_ball_dx_n=self._last_ball_dx_n,
            last_ball_dy_n=self._last_ball_dy_n,
        )

        # Update last-seen ball state only when currently visible
        if base_obs[8] > 0.0:  # ball_visible_n == +1
            self._last_ball_x_n = float(base_obs[4])
            self._last_ball_y_n = float(base_obs[5])
            self._last_ball_dx_n = float(base_obs[6])
            self._last_ball_dy_n = float(base_obs[7])

        # Optionally clip to the target range to avoid rare excursions at borders
        if self.clip:
            base_obs = np.clip(base_obs, -1.0, 1.0, out=base_obs)

        obs = np.concatenate((base_obs, self._last_action_one_hot), dtype=np.float32)

        # Always ensure finiteness
        assert np.all(np.isfinite(obs)), f"Non-finite values found: {obs}"
        return obs
