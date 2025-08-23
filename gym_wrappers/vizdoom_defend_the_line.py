import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class VizDoomDefendTheLineEnv(gym.Env):
    """Minimal Gymnasium wrapper around ViZDoom Defend The Line scenario.

    This environment expects ViZDoom to be installed (pip install vizdoom).

    Parameters (via env_kwargs):
        config_path: Optional[str] - Path to defend_the_line.cfg. If not provided,
            attempts to locate the scenarios directory in the installed vizdoom
            package or the VIZDOOM_SCENARIOS_DIR environment variable.
        render_mode: Optional[str] - 'rgb_array' (default) or 'human'.
        seed: Optional[int] - Seed for ViZDoom RNG.
        frame_skip: Optional[int] - Number of frames to repeat each action (defaults to 1).
    """

    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 35,
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        frame_skip: int = 1,
        **_: Dict[str, Any],
    ) -> None:
        super().__init__()

        try:
            import vizdoom as vzd
        except Exception as exc:  # pragma: no cover - import-time dependency
            raise ImportError(
                "vizdoom package is required for VizDoomDefendTheLineEnv. "
                "Install with: pip install vizdoom"
            ) from exc

        self._vzd = vzd
        self._render_mode = render_mode or "rgb_array"
        self._frame_skip = max(1, int(frame_skip))
        self._game = vzd.DoomGame()

        # Resolve config path
        cfg_path = self._resolve_config_path(config_path)
        if cfg_path is None:
            raise FileNotFoundError(
                "Could not locate defend_the_line.cfg. Set env_kwargs.config_path "
                "or define VIZDOOM_SCENARIOS_DIR to the directory containing the scenario files."
            )

        self._game.load_config(str(cfg_path))

        # Configure visibility based on render mode
        self._game.set_window_visible(self._render_mode == "human")

        if seed is not None:
            try:
                self._game.set_seed(int(seed))
            except Exception:
                pass

        # Ensure RGB24 for predictable observations (cfg typically sets it already)
        try:
            self._game.set_screen_format(vzd.ScreenFormat.RGB24)
        except Exception:
            pass

        self._game.init()

        # Determine observation and action spaces from the initialized game
        height = int(self._game.get_screen_height())
        width = int(self._game.get_screen_width())
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )

        # Build a small discrete action set mapped to vizdoom button vectors
        n_buttons = int(self._game.get_available_buttons_size())
        # Helper to build binary vectors by button indices
        def a(*on_idx: int):
            vec = [0] * n_buttons
            for i in on_idx:
                if 0 <= i < n_buttons:
                    vec[i] = 1
            return vec
        # Common minimal controls. Index order follows vizdoom available buttons
        # Typical order: MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ATTACK
        self._discrete_actions = [
            a(),      # 0: noop
            a(2),     # 1: forward
            a(3),     # 2: backward
            a(0),     # 3: strafe left
            a(1),     # 4: strafe right
            a(4),     # 5: turn left
            a(5),     # 6: turn right
            a(6),     # 7: attack
            a(2, 6),  # 8: forward + attack
        ]
        self.action_space = gym.spaces.Discrete(len(self._discrete_actions))

        self._last_obs: Optional[np.ndarray] = None

    def _resolve_config_path(self, config_path: Optional[str]) -> Optional[Path]:
        if config_path:
            candidate = Path(config_path)
            if candidate.is_file():
                return candidate
        # Try environment variable first
        env_dir = os.environ.get("VIZDOOM_SCENARIOS_DIR")
        if env_dir:
            p = Path(env_dir) / "defend_the_line.cfg"
            if p.is_file():
                return p
        # Try to locate scenarios folder inside the installed vizdoom package
        try:
            import vizdoom as vzd  # type: ignore
            pkg_dir = Path(vzd.__file__).parent
            scenarios_dir = pkg_dir / "scenarios"
            p = scenarios_dir / "defend_the_line.cfg"
            if p.is_file():
                return p
        except Exception:
            pass
        return None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            try:
                self._game.set_seed(int(seed))
            except Exception:
                pass
        self._game.new_episode()
        obs = self._get_screen()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map Discrete action index to a vizdoom button vector
        try:
            idx = int(action) if not isinstance(action, (list, tuple, np.ndarray)) else int(np.asarray(action).item())
        except Exception:
            idx = int(action)
        idx = max(0, min(idx, len(self._discrete_actions) - 1))
        action_list = self._discrete_actions[idx]

        reward = float(self._game.make_action(action_list, self._frame_skip))

        terminated = self._game.is_episode_finished()
        truncated = False

        if terminated:
            obs = self.observation_space.low  # type: ignore[assignment]
        else:
            obs = self._get_screen()

        info: Dict[str, Any] = {}
        try:
            info["health"] = float(self._game.get_game_variable(self._vzd.GameVariable.HEALTH))
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._render_mode == "rgb_array":
            return self._get_screen()
        return None

    @property
    def render_mode(self) -> Optional[str]:
        return self._render_mode

    def close(self) -> None:
        try:
            self._game.close()
        except Exception:
            pass

    # Helpers
    def _get_screen(self) -> np.ndarray:
        state = self._game.get_state()
        if state is None or state.screen_buffer is None:
            # When episode finished, there is no valid frame; return zeros
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        # ViZDoom returns (C,H,W) by default; transpose to (H,W,C)
        screen = state.screen_buffer
        if screen.ndim == 3 and screen.shape[0] in (1, 3):
            screen = np.moveaxis(screen, 0, -1)
        screen = np.ascontiguousarray(screen)
        self._last_obs = screen
        return screen


