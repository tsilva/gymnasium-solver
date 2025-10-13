import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class VizDoomEnv(gym.Env):
    """Generic Gymnasium wrapper around ViZDoom scenarios.

    Parameters (via env_kwargs):
        scenario: Optional[str] - One of {"basic", "deadly_corridor", "deathmatch",
            "defend_the_center", "defend_the_line", "health_gathering"}. If provided,
            the wrapper will try to locate the corresponding .cfg file from either
            VIZDOOM_SCENARIOS_DIR or the installed vizdoom package. Ignored if config_path is provided.
        config_path: Optional[str] - Path to a specific scenario .cfg file.
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
        scenario: Optional[str] = None,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        frame_skip: int = 1,
        **_: Dict[str, Any],
    ) -> None:
        super().__init__()

        try:
            import vizdoom as vzd
        except ImportError as exc:  # pragma: no cover - import-time dependency
            raise ImportError(
                "vizdoom package is required for VizDoomEnv. Install with: pip install vizdoom"
            ) from exc

        self._vzd = vzd
        self._render_mode = render_mode or "rgb_array"
        self._frame_skip = max(1, int(frame_skip))
        self._game = vzd.DoomGame()

        # Resolve config path
        cfg_path = self._resolve_config_path(config_path=config_path, scenario=scenario)
        if cfg_path is None:
            hint = scenario or "<unknown>"
            raise FileNotFoundError(
                f"Could not locate scenario cfg for '{hint}'. Set env_kwargs.config_path or define "
                "VIZDOOM_SCENARIOS_DIR to the directory containing the scenario files."
            )

        # For scenarios requiring doom.wad, locate it before loading config
        if scenario and scenario.lower() in ("e1m1",):
            doom_wad_path = self._find_doom_wad()
            if doom_wad_path:
                self._game.set_doom_game_path(str(doom_wad_path))
            else:
                raise FileNotFoundError(
                    f"doom.wad not found for scenario '{scenario}'. "
                    "Please obtain doom.wad (from Steam/GOG) or use FreeDoom, and place it in one of:\n"
                    f"  1. VizDoom scenarios dir: {Path(self._vzd.__file__).parent / 'scenarios'}\n"
                    f"  2. VIZDOOM_SCENARIOS_DIR env var location\n"
                    f"  3. Current working directory: {Path.cwd()}\n"
                    "See vizdoom_configs/README.md for detailed setup instructions."
                )

        self._game.load_config(str(cfg_path))

        # Configure visibility based on render mode
        self._game.set_window_visible(self._render_mode == "human")

        if seed is not None:
            self._game.set_seed(int(seed))

        # Ensure RGB24 for predictable observations (cfg typically sets it already)
        self._game.set_screen_format(vzd.ScreenFormat.RGB24)

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

        def _action_vector(*on_idx: int):
            vector = [0] * n_buttons
            for i in on_idx:
                if 0 <= i < n_buttons:
                    vector[i] = 1
            return vector

        # Common minimal controls. Index order follows vizdoom available buttons
        # Typical order: MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, ATTACK
        self._discrete_actions = [
            _action_vector(),      # 0: noop
            _action_vector(2),     # 1: forward
            _action_vector(3),     # 2: backward
            _action_vector(0),     # 3: strafe left
            _action_vector(1),     # 4: strafe right
            _action_vector(4),     # 5: turn left
            _action_vector(5),     # 6: turn right
            _action_vector(6),     # 7: attack
            #_action_vector(2, 6),  # 8: forward + attack
        ]
        self.action_space = gym.spaces.Discrete(len(self._discrete_actions))

        self._last_obs: Optional[np.ndarray] = None

    @property
    def render_mode(self) -> Optional[str]:
        return self._render_mode

    def _resolve_config_path(self, config_path: Optional[str], scenario: Optional[str]) -> Optional[Path]:
        if config_path:
            candidate = Path(config_path)
            if candidate.is_file():
                return candidate

        # Choose default cfg file name by scenario
        cfg_by_scenario = {
            "basic": "basic.cfg",
            "deadly_corridor": "deadly_corridor.cfg",
            "deathmatch": "deathmatch.cfg",
            "defend_the_center": "defend_the_center.cfg",
            "defend_the_line": "defend_the_line.cfg",
            "health_gathering": "health_gathering.cfg",
            "my_way_home": "my_way_home.cfg",
            "e1m1": "doom_e1m1.cfg",
        }

        cfg_name = None
        key = (scenario or "").strip().lower().replace(" ", "_")
        if key in cfg_by_scenario:
            cfg_name = cfg_by_scenario[key]

        # For custom configs (like e1m1), check project vizdoom_configs directory first
        if cfg_name and key in ("e1m1",):
            project_configs = Path(__file__).parent.parent / "vizdoom_configs" / cfg_name
            if project_configs.is_file():
                return project_configs

        # Try environment variable first
        if cfg_name:
            env_dir = os.environ.get("VIZDOOM_SCENARIOS_DIR")
            if env_dir:
                p = Path(env_dir) / cfg_name
                if p.is_file():
                    return p

        # Try to locate scenarios folder inside the installed vizdoom package
        # Use the already-imported vizdoom module
        pkg_dir = Path(self._vzd.__file__).parent
        scenarios_dir = pkg_dir / "scenarios"
        if cfg_name:
            p = scenarios_dir / cfg_name
            if p.is_file():
                return p

        return None

    def _find_doom_wad(self) -> Optional[Path]:
        """Search for doom.wad in standard locations."""
        # Check VIZDOOM_SCENARIOS_DIR
        env_dir = os.environ.get("VIZDOOM_SCENARIOS_DIR")
        if env_dir:
            p = Path(env_dir) / "doom.wad"
            if p.is_file():
                return p

        # Check installed vizdoom package scenarios directory
        pkg_dir = Path(self._vzd.__file__).parent
        p = pkg_dir / "scenarios" / "doom.wad"
        if p.is_file():
            return p

        # Check current working directory
        p = Path.cwd() / "doom.wad"
        if p.is_file():
            return p

        return None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._game.set_seed(int(seed))
        self._game.new_episode()
        obs = self._get_screen()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map Discrete action index to a vizdoom button vector
        idx = int(action) if not isinstance(action, (list, tuple, np.ndarray)) else int(np.asarray(action).item())
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
        # Expose common game variables (fail fast if unavailable)
        info["health"] = float(self._game.get_game_variable(self._vzd.GameVariable.HEALTH))
        info["ammo"] = float(self._game.get_game_variable(self._vzd.GameVariable.AMMO2))
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._render_mode == "rgb_array":
            return self._get_screen()
        return None


    def close(self) -> None:
        self._game.close()

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
