from __future__ import annotations

import math
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium.vector import VectorWrapper
from gymnasium import spaces

# TODO: CLEANUP this file

def _term_clear_inplace() -> None:
    # Clear screen and move cursor to home (top-left)
    # Avoid if output is not a TTY to reduce log noise
    if os.getenv("NO_COLOR", "").lower() == "1":
        return
    try:
        is_tty = os.isatty(1)
    except Exception:
        is_tty = False
    if not is_tty:
        return
    print("\x1b[2J\x1b[H", end="")


def _parse_inf(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"inf", "+inf", "infinity", "+infinity", "∞"}:
            return float("inf")
        if s in {"-inf", "-infinity", "-∞"}:
            return float("-inf")
        try:
            return float(s)
        except Exception:
            return None
    return None


def _get_terminal_width(default: int = 80) -> int:
    try:
        cols, _ = shutil.get_terminal_size(fallback=(default, 24))
        return int(cols)
    except Exception:
        return default


class VecObsBarPrinter(VectorWrapper):
    """
    VectorEnv wrapper that prints a per-dimension observation table with bars.

    - Uses environment spec (YAML) to label observation components when available.
    - If the spec provides finite ranges, uses them; otherwise, tracks running
      min/max to scale bars dynamically.
    - Clears the terminal and prints in place each step for an easy live view.

    Notes:
    - Designed for 1D (vector) observations. For non-vector observations
      (e.g., images), output is suppressed.
    - Intended primarily for play/debugging; attach only for human render runs.
    """

    def __init__(
        self,
        env,
        *,
        bar_width: int = 40,
        env_index: int = 0,
        enable: bool = True,
        target_episodes: int,
    ) -> None:
        super().__init__(env)
        self._bar_width = max(10, int(bar_width))
        self._env_index = int(env_index)
        self._enable = bool(enable)
        target_episodes_int = int(target_episodes)
        assert target_episodes_int >= 1, "target_episodes must be >= 1"
        self._target_episodes: int = target_episodes_int

        # Labels and ranges derived from spec, when available
        self._labels: Optional[List[str]] = None
        self._ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
        # Reward range from spec (lo, hi) when available
        self._reward_range: Tuple[Optional[float], Optional[float]] = (None, None)
        # Action metadata (from spec or action_space)
        self._action_discrete_n: Optional[int] = None
        self._action_labels: Optional[Dict[int, str]] = None
        self._last_actions: Optional[np.ndarray] = None
        self._action_probs: Optional[np.ndarray] = None
        # Cached time limit (max episode steps), if available via env info
        self._time_limit: Optional[int] = None

        # Dynamic scaling when finite ranges are not available
        self._min_seen: Optional[np.ndarray] = None
        self._max_seen: Optional[np.ndarray] = None

        # Lazily detect vector obs and initialize from first observation
        self._initialized = False

        # Episode accounting for the chosen env index
        self._ep_count: int = 0
        self._current_ep_return: float = 0.0
        self._current_ep_len: int = 0
        self._last_ep_return: Optional[float] = None
        self._last_ep_len: Optional[int] = None
        # Running mean over completed episodes in this session
        self._sum_ep_returns: float = 0.0

        # Try reading spec to pre-populate labels/ranges
        self._init_from_spec()

    # ---- Public methods ----
    def set_action_probs(self, action_probs: np.ndarray) -> None:
        """Set action probabilities for visualization."""
        self._action_probs = action_probs

    # ---- VectorEnv overrides ----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._maybe_init_from_obs(obs)
        # Reset episode tracking on full env reset
        self._ep_count = 0
        self._current_ep_return = 0.0
        self._current_ep_len = 0
        self._last_ep_return = None
        self._last_ep_len = None
        self._sum_ep_returns = 0.0
        # Print initial observation if enabled
        if self._enable:
            self._print_obs(obs, rewards=None, dones=None)
        return obs, info

    def step(self, actions):
        # Cache last actions for rendering (best-effort; shapes vary by env)
        try:
            self._last_actions = np.asarray(actions)
        except Exception:
            self._last_actions = None

        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        self._maybe_init_from_obs(obs)

        # Combine terminated and truncated into dones for internal tracking
        dones = np.logical_or(terminated, truncated)

        # Update episode accounting for the selected env index
        try:
            if isinstance(rewards, np.ndarray) and rewards.size > 0:
                self._current_ep_return += float(rewards[self._env_index])
                self._current_ep_len += 1
            if isinstance(dones, np.ndarray) and dones.size > 0 and bool(dones[self._env_index]):
                self._last_ep_return = self._current_ep_return
                self._last_ep_len = self._current_ep_len
                self._ep_count += 1
                self._sum_ep_returns += self._current_ep_return
                self._current_ep_return = 0.0
                self._current_ep_len = 0
        except Exception:
            pass

        if self._enable:
            self._print_obs(obs, rewards=rewards, dones=dones)
        return obs, rewards, terminated, truncated, infos

    # ---- Internal helpers ----
    def _init_from_spec(self) -> None:
        # VecEnvInfoWrapper provides get_spec(); all envs must have it
        spec = self.env.get_spec()

        # Parse reward range and action metadata first (independent of observation space)
        self._parse_reward_range_from_spec(spec)
        self._parse_action_metadata_from_spec(spec)

        # Extract observation components from spec
        if spec.observation_space is None:
            return
        obs_space = spec.observation_space
        if obs_space.default is None:
            return
        variant = obs_space.variants.get(obs_space.default)
        if variant is None or not variant.components:
            return

        labels: List[str] = []
        ranges: List[Tuple[Optional[float], Optional[float]]] = []
        for comp in variant.components:
            if comp.name is None:
                continue
            labels.append(comp.name)
            # Extract range from component
            if comp.range is not None:
                ranges.append((float(comp.range[0]), float(comp.range[1])))
            elif comp.values:
                try:
                    ranges.append((float(min(comp.values)), float(max(comp.values))))
                except (ValueError, TypeError):
                    ranges.append((None, None))
            else:
                ranges.append((None, None))

        if labels:
            self._labels = labels
            self._ranges = ranges if len(ranges) == len(labels) else None

    def _parse_reward_range_from_spec(self, spec) -> None:
        rr = spec.get_reward_range()
        if rr is not None and len(rr) == 2:
            lo = _parse_inf(rr[0])
            hi = _parse_inf(rr[1])
            self._reward_range = (lo, hi)

    def _parse_action_metadata_from_spec(self, spec) -> None:
        if spec.action_space is not None:
            if spec.action_space.discrete is not None:
                self._action_discrete_n = int(spec.action_space.discrete)

        action_labels = spec.get_action_labels()
        if action_labels:
            parsed: Dict[int, str] = {}
            for k, v in action_labels.items():
                if isinstance(k, int):
                    parsed[k] = str(v)
                elif isinstance(k, str):
                    try:
                        parsed[int(k)] = str(v)
                    except ValueError:
                        pass
            if parsed:
                self._action_labels = parsed

    def _maybe_init_from_obs(self, obs: np.ndarray) -> None:
        if self._initialized:
            return
        # Expect obs shape (n_envs, dim) for vector observations
        try:
            if not isinstance(obs, np.ndarray) or obs.ndim < 2:
                # Not a vector observation; do not initialize
                self._initialized = True
                return
            n_envs = int(obs.shape[0])
            if self._env_index < 0 or self._env_index >= n_envs:
                self._env_index = 0
            # Flatten the rest of the dims; suppress for high-D (e.g., images)
            rest_shape = obs.shape[1:]
            dim = int(np.prod(rest_shape))
            if dim <= 0:
                self._initialized = True
                return
            # Note: For non-vector observations (images/stacked frames),
            # we still enable printing but skip observation bars in _print_obs
            # Initialize dynamic range trackers
            self._min_seen = np.full((dim,), np.inf, dtype=np.float64)
            self._max_seen = np.full((dim,), -np.inf, dtype=np.float64)
        finally:
            self._initialized = True

    def _labels_for_dim(self, dim: int) -> List[str]:
        if isinstance(self._labels, list) and len(self._labels) == dim:
            return self._labels
        # Fallback generic labels
        return [f"obs[{i}]" for i in range(dim)]

    def _ranges_for_dim(self, dim: int) -> List[Tuple[Optional[float], Optional[float]]]:
        if isinstance(self._ranges, list) and len(self._ranges) == dim:
            return self._ranges
        return [(None, None) for _ in range(dim)]

    def _normalize_range(self, lo: Optional[float], hi: Optional[float]) -> Tuple[float, float]:
        """Normalize a range to valid finite bounds, defaulting to [-1, 1]."""
        lo_eff = float(lo) if (lo is not None and math.isfinite(lo)) else None
        hi_eff = float(hi) if (hi is not None and math.isfinite(hi)) else None
        if lo_eff is None or hi_eff is None or hi_eff == lo_eff:
            return -1.0, 1.0
        return lo_eff, hi_eff

    def _draw_bar(self, value: float, lo_eff: float, hi_eff: float, width: int) -> Tuple[str, float]:
        """Draw a bar with given value and effective range."""
        ratio = 0.0 if hi_eff == lo_eff else (float(value) - lo_eff) / (hi_eff - lo_eff)
        ratio = max(0.0, min(1.0, ratio))
        filled = int(round(ratio * width))
        empty = width - filled
        bar = "█" * filled + "·" * empty
        return bar, ratio

    def _format_bar(self, value: float, lo: Optional[float], hi: Optional[float], i: int) -> Tuple[str, float, Tuple[float, float]]:
        # Update dynamic ranges when finite bounds are not available
        if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
            if self._min_seen is not None and self._max_seen is not None:
                self._min_seen[i] = min(self._min_seen[i], float(value))
                self._max_seen[i] = max(self._max_seen[i], float(value))
                lo, hi = float(self._min_seen[i]), float(self._max_seen[i])
        lo_eff, hi_eff = self._normalize_range(lo, hi)
        bar, ratio = self._draw_bar(value, lo_eff, hi_eff, self._bar_width)
        return bar, ratio, (lo_eff, hi_eff)

    def _format_bar_scalar(self, value: float, lo: Optional[float], hi: Optional[float], *, width: Optional[int] = None) -> Tuple[str, float, Tuple[float, float]]:
        """Format a bar for a single scalar value using provided bounds only."""
        lo_eff, hi_eff = self._normalize_range(lo, hi)
        bw = self._bar_width if width is None else max(1, int(width))
        bar, ratio = self._draw_bar(value, lo_eff, hi_eff, bw)
        return bar, ratio, (lo_eff, hi_eff)

    def _print_bar_line(self, label: str, value_str: str, bar: str, lo_eff: float, hi_eff: float, label_w: int, value_w: int) -> None:
        """Print a formatted bar line with label, value, bar, and range."""
        label_fmt = label[:label_w].ljust(label_w)
        val_fmt = value_str.rjust(value_w)
        rng_fmt = f"[{lo_eff:+.2f}, {hi_eff:+.2f}]"
        print(f"{label_fmt}  {val_fmt}  {bar}  {rng_fmt}")

    def _extract_env_value(self, arr: Optional[np.ndarray]) -> Optional[float]:
        """Extract value from env-indexed array."""
        try:
            return float(arr[self._env_index]) if isinstance(arr, np.ndarray) and arr.size > 0 else None
        except Exception:
            return None

    def _extract_action_index(self) -> Optional[int]:
        """Extract action index from last actions array (for Discrete action spaces)."""
        try:
            if self._last_actions is not None:
                arr = np.asarray(self._last_actions)
                if arr.ndim == 0:
                    return int(arr.item())
                elif arr.size > self._env_index:
                    val = arr[self._env_index]
                    # Handle both scalar and array values
                    if isinstance(val, (int, np.integer)):
                        return int(val)
                    elif isinstance(val, np.ndarray):
                        return int(val.flatten()[0])
                    else:
                        return int(val)
        except Exception:
            pass
        return None

    def _extract_multibinary_action(self) -> Optional[np.ndarray]:
        """Extract action vector from last actions array (for MultiBinary action spaces)."""
        try:
            if self._last_actions is not None:
                arr = np.asarray(self._last_actions)
                # For MultiBinary, expect shape (n_envs, n_actions)
                if arr.ndim >= 2 and arr.shape[0] > self._env_index:
                    return arr[self._env_index]
                # If shape is (n_actions,), assume single env
                elif arr.ndim == 1:
                    return arr
        except Exception:
            pass
        return None

    def _print_obs(
        self,
        obs_full: np.ndarray,
        *,
        rewards: Optional[np.ndarray],
        dones: Optional[np.ndarray],
    ) -> None:
        if not isinstance(obs_full, np.ndarray) or obs_full.ndim < 2:
            return
        # Extract single env observation
        obs = obs_full[self._env_index]
        if obs is None:
            return
        obs_arr = np.asarray(obs)

        # Check if observation is 1D vector (for obs bars later)
        is_vector_obs = obs_arr.ndim == 1
        dim = int(obs_arr.shape[0]) if is_vector_obs else 0
        labels = self._labels_for_dim(dim) if is_vector_obs else []
        ranges = self._ranges_for_dim(dim) if is_vector_obs else []

        # Prepare header and lines
        _term_clear_inplace()

        term_w = _get_terminal_width()
        label_w = max(12, min(28, max(len(s) for s in labels) if labels else 12))
        value_w = 10  # width for numeric value
        bar_w = max(10, min(self._bar_width, term_w - label_w - value_w - 10))
        self._bar_width = bar_w

        # Optional status line (only done flag now; reward shown below as a bar)
        try:
            is_done = isinstance(dones, np.ndarray) and dones.size > 0 and bool(dones[self._env_index])
        except Exception:
            is_done = False
        status = "done=True" if is_done else ""

        env_id = self.env.get_id() if hasattr(self.env, "get_id") else None

        # Build header with episode info, last reward, and running mean reward
        curr_ep_idx = self._ep_count + 1  # 1-based index for current episode
        ep_prog = f"Ep {curr_ep_idx}/{self._target_episodes}"

        # Current episode progress (return and length)
        parts = [
            env_id or "",
            ep_prog,
            f"cur_len={int(self._current_ep_len)}",
            f"cur_ep_r={self._current_ep_return:+.3f}",
            f"last_ep_r={self._last_ep_return:+.3f}" if isinstance(self._last_ep_return, (int, float)) else "last_ep_r=--",
            f"ep_rew_mean={(self._sum_ep_returns / self._ep_count):+.3f}" if isinstance(self._ep_count, int) and self._ep_count > 0 else "ep_rew_mean=--"
        ]
        header_main = "  ".join(x for x in parts if x)
        print(header_main)
        if status:
            print(status)

        # Episode timestep progress bar, if time limit is known
        if self._time_limit is None:
            self._time_limit = self.env.get_max_episode_steps()
        if self._time_limit:
            steps = int(self._current_ep_len)
            total = int(self._time_limit)
            t_bar, _t_ratio, (t_lo_eff, t_hi_eff) = self._format_bar_scalar(steps, 0, total, width=self._bar_width)
            self._print_bar_line("timestep", f"{steps}/{total}", t_bar, t_lo_eff, t_hi_eff, label_w, value_w)

        print("─" * (label_w + value_w + bar_w + 24))

        # Reward bar (scaled using spec reward range when available)
        reward_val = self._extract_env_value(rewards)
        if reward_val is not None:
            r_lo, r_hi = self._reward_range
            r_bar, _r_ratio, (r_lo_eff, r_hi_eff) = self._format_bar_scalar(reward_val, r_lo, r_hi, width=self._bar_width)
            self._print_bar_line("reward", f"{reward_val:+.4f}", r_bar, r_lo_eff, r_hi_eff, label_w, value_w)
            print("─" * (label_w + value_w + bar_w + 24))

        # Action bars: show probabilities/values for each action
        action_idx = self._extract_action_index()
        multibinary_action = self._extract_multibinary_action()

        # For VectorEnv, use single_action_space
        action_space = getattr(self, 'single_action_space', self.action_space)
        is_multibinary = isinstance(action_space, spaces.MultiBinary)

        # Determine number of actions based on action space type
        if isinstance(action_space, spaces.Discrete):
            act_n = int(action_space.n)
        elif is_multibinary:
            act_n = int(action_space.n)  # Number of binary actions/buttons
        elif isinstance(self._action_discrete_n, int) and self._action_discrete_n > 0:
            act_n = self._action_discrete_n
        else:
            act_n = None

        # Display action info
        if act_n is not None and act_n > 0:
            if self._action_probs is not None and len(self._action_probs) > self._env_index:
                # Show one bar per action with probability
                probs = self._action_probs[self._env_index]
                for a_idx in range(min(act_n, len(probs))):
                    prob = float(probs[a_idx])
                    a_bar, _a_ratio, (a_lo_eff, a_hi_eff) = self._format_bar_scalar(prob, 0.0, 1.0, width=self._bar_width)
                    action_label = self._action_labels.get(a_idx) if self._action_labels else None
                    label = f"a[{a_idx}]" + (f" {action_label}" if action_label else "")
                    val_str = f"{prob:.4f}"
                    self._print_bar_line(label, val_str, a_bar, a_lo_eff, a_hi_eff, label_w, value_w)
            elif is_multibinary and multibinary_action is not None:
                # MultiBinary: show one bar per button with 0 or 1 value
                for a_idx in range(min(act_n, len(multibinary_action))):
                    binary_val = float(multibinary_action[a_idx])
                    a_bar, _a_ratio, (a_lo_eff, a_hi_eff) = self._format_bar_scalar(binary_val, 0.0, 1.0, width=self._bar_width)
                    action_label = self._action_labels.get(a_idx) if self._action_labels else None
                    label = f"a[{a_idx}]" + (f" {action_label}" if action_label else "")
                    val_str = f"{int(binary_val)}"  # Show as 0 or 1 (integer)
                    self._print_bar_line(label, val_str, a_bar, a_lo_eff, a_hi_eff, label_w, value_w)
            elif action_idx is not None:
                # Discrete: show one bar per action with binary indicator (1.0 for selected, 0.0 for others)
                for a_idx in range(act_n):
                    binary_val = 1.0 if a_idx == action_idx else 0.0
                    a_bar, _a_ratio, (a_lo_eff, a_hi_eff) = self._format_bar_scalar(binary_val, 0.0, 1.0, width=self._bar_width)
                    action_label = self._action_labels.get(a_idx) if self._action_labels else None
                    label = f"a[{a_idx}]" + (f" {action_label}" if action_label else "")
                    val_str = f"{binary_val:.1f}"
                    self._print_bar_line(label, val_str, a_bar, a_lo_eff, a_hi_eff, label_w, value_w)
            else:
                # Fallback: show action space size when we can't extract the specific action
                space_type = "multibinary" if is_multibinary else "discrete"
                print(f"action: ? ({space_type}: {act_n})")

        print("─" * (label_w + value_w + bar_w + 24))

        # Only print observation bars for vector observations
        if is_vector_obs:
            for i in range(dim):
                label_suffix = labels[i] if i < len(labels) else None
                name = f"o[{i}]" + (f" {label_suffix}" if label_suffix else "")
                val = float(obs_arr[i])
                lo, hi = ranges[i] if i < len(ranges) else (None, None)
                bar, _ratio, (lo_eff, hi_eff) = self._format_bar(val, lo, hi, i)
                self._print_bar_line(name, f"{val:+.4f}", bar, lo_eff, hi_eff, label_w, value_w)
        else:
            # For non-vector obs (images), just note the shape
            print(f"[image observation: shape={obs_arr.shape}]")
