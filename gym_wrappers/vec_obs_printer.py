from __future__ import annotations

import math
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper


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
        # Normalize string-y NaN/Inf already parsed to float
        if math.isfinite(float(x)):
            return float(x)
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"inf", "+inf", "infinity", "+infinity", "∞"}:
            return float("inf")
        if s in {"-inf", "-infinity", "-∞"}:
            return float("-inf")
        # Best-effort float parse
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


class VecObsBarPrinter(VecEnvWrapper):
    """
    VecEnv wrapper that prints a per-dimension observation table with bars.

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
        venv,
        *,
        bar_width: int = 40,
        env_index: int = 0,
        enable: bool = True,
        target_episodes: Optional[int] = None,
    ) -> None:
        super().__init__(venv)
        self._bar_width = max(10, int(bar_width))
        self._env_index = int(env_index)
        self._enable = bool(enable)
        self._target_episodes = int(target_episodes) if target_episodes is not None else None

        # Labels and ranges derived from spec, when available
        self._labels: Optional[List[str]] = None
        self._ranges: Optional[List[Tuple[Optional[float], Optional[float]]]] = None

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

        # Try reading spec to pre-populate labels/ranges
        self._init_from_spec()

    # ---- VecEnv overrides ----
    def reset(self):
        obs = self.venv.reset()
        self._maybe_init_from_obs(obs)
        # Reset episode tracking on full env reset
        self._ep_count = 0
        self._current_ep_return = 0.0
        self._current_ep_len = 0
        self._last_ep_return = None
        self._last_ep_len = None
        # Print initial observation if enabled
        if self._enable:
            self._print_obs(obs, rewards=None, dones=None)
        return obs

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._maybe_init_from_obs(obs)

        # Update episode accounting for the selected env index
        try:
            if isinstance(rewards, np.ndarray) and rewards.size > 0:
                r = float(rewards[self._env_index])
                self._current_ep_return += r
                self._current_ep_len += 1
            if isinstance(dones, np.ndarray) and dones.size > 0 and bool(dones[self._env_index]):
                self._last_ep_return = float(self._current_ep_return)
                self._last_ep_len = int(self._current_ep_len)
                self._ep_count += 1
                self._current_ep_return = 0.0
                self._current_ep_len = 0
        except Exception:
            pass

        if self._enable:
            self._print_obs(obs, rewards=rewards, dones=dones)
        return obs, rewards, dones, infos

    # ---- Internal helpers ----
    def _init_from_spec(self) -> None:
        # VecEnvInfoWrapper provides get_spec(); if missing, we skip labels
        spec: Optional[Dict[str, Any]] = None
        try:
            if hasattr(self.venv, "get_spec"):
                spec = self.venv.get_spec()
        except Exception:
            spec = None
        if not isinstance(spec, dict):
            return

        obs_space = spec.get("observation_space")
        if not isinstance(obs_space, dict):
            return
        default_name = obs_space.get("default")
        variants = obs_space.get("variants", {})
        variant = variants.get(default_name) if isinstance(variants, dict) else None
        if not isinstance(variant, dict):
            return
        components = variant.get("components")
        if not isinstance(components, Sequence):
            return

        labels: List[str] = []
        ranges: List[Tuple[Optional[float], Optional[float]]] = []
        for comp in components:
            if not isinstance(comp, dict):
                continue
            name = comp.get("name")
            if not isinstance(name, str):
                continue
            labels.append(str(name))
            # Prefer explicit numeric range; otherwise, derive from values
            lo, hi = None, None
            if "range" in comp:
                rng = comp.get("range")
                if isinstance(rng, Sequence) and len(rng) == 2:
                    lo = _parse_inf(rng[0])
                    hi = _parse_inf(rng[1])
            elif "values" in comp:
                vals = comp.get("values")
                if isinstance(vals, Sequence) and len(vals) > 0:
                    try:
                        min_v = min(float(v) for v in vals)
                        max_v = max(float(v) for v in vals)
                        lo, hi = float(min_v), float(max_v)
                    except Exception:
                        lo, hi = None, None
            ranges.append((lo, hi))

        if labels:
            self._labels = labels
            self._ranges = ranges if len(ranges) == len(labels) else None

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
            if len(rest_shape) != 1:
                # Likely an image or stacked frames; disable output
                self._enable = False
                self._initialized = True
                return
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

    def _format_bar(self, value: float, lo: Optional[float], hi: Optional[float], i: int) -> Tuple[str, float, Tuple[float, float]]:
        # Update dynamic ranges when finite bounds are not available
        if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
            if self._min_seen is not None and self._max_seen is not None:
                # Track running min/max for dynamic scaling
                self._min_seen[i] = min(self._min_seen[i], float(value))
                self._max_seen[i] = max(self._max_seen[i], float(value))
                lo_eff = float(self._min_seen[i])
                hi_eff = float(self._max_seen[i])
            else:
                lo_eff, hi_eff = -1.0, 1.0
        else:
            lo_eff, hi_eff = float(lo), float(hi)

        # Avoid zero-width ranges
        if not math.isfinite(lo_eff) or not math.isfinite(hi_eff) or hi_eff == lo_eff:
            lo_eff, hi_eff = -1.0, 1.0

        # Compute fill ratio
        ratio = 0.0 if hi_eff == lo_eff else (float(value) - lo_eff) / (hi_eff - lo_eff)
        ratio = max(0.0, min(1.0, ratio))
        filled = int(round(ratio * self._bar_width))
        empty = self._bar_width - filled
        bar = "█" * filled + "·" * empty
        return bar, ratio, (lo_eff, hi_eff)

    def _print_obs(
        self,
        obs_full: np.ndarray,
        *,
        rewards: Optional[np.ndarray],
        dones: Optional[np.ndarray],
    ) -> None:
        if not isinstance(obs_full, np.ndarray) or obs_full.ndim < 2:
            return
        # Extract single env observation as a flat 1D vector
        obs = obs_full[self._env_index]
        if obs is None:
            return
        obs_arr = np.asarray(obs)
        if obs_arr.ndim != 1:
            # Non-vector obs (e.g., images) → skip
            return

        dim = int(obs_arr.shape[0])
        labels = self._labels_for_dim(dim)
        ranges = self._ranges_for_dim(dim)

        # Prepare header and lines
        _term_clear_inplace()

        term_w = _get_terminal_width()
        label_w = max(12, min(28, max(len(s) for s in labels) if labels else 12))
        value_w = 10  # width for numeric value
        bar_w = max(10, min(self._bar_width, term_w - label_w - value_w - 10))
        self._bar_width = bar_w

        # Optional status line from rewards/dones
        status_parts: List[str] = []
        if isinstance(rewards, np.ndarray) and rewards.size > 0:
            try:
                r = float(rewards[self._env_index])
                status_parts.append(f"r={r:+.3f}")
            except Exception:
                pass
        if isinstance(dones, np.ndarray) and dones.size > 0:
            try:
                d = bool(dones[self._env_index])
                if d:
                    status_parts.append("done=True")
            except Exception:
                pass
        status = "  ".join(status_parts)

        env_id = None
        try:
            if hasattr(self.venv, "get_id"):
                env_id = self.venv.get_id()
        except Exception:
            env_id = None

        # Build header with episode info and last episode reward
        curr_ep_idx = self._ep_count + 1  # 1-based index for current episode
        if self._target_episodes is not None and self._target_episodes > 0:
            ep_prog = f"Ep {curr_ep_idx}/{self._target_episodes}"
        else:
            ep_prog = f"Ep {curr_ep_idx}"
        last_ep = (
            f"last_ep_r={self._last_ep_return:+.3f}"
            if isinstance(self._last_ep_return, (int, float))
            else "last_ep_r=--"
        )
        header_main = "  ".join(x for x in [env_id or "", ep_prog, last_ep] if x)
        print(header_main)
        if status:
            print(status)

        for i in range(dim):
            name = labels[i] if i < len(labels) else f"obs[{i}]"
            val = float(obs_arr[i])
            lo, hi = ranges[i] if i < len(ranges) else (None, None)
            bar, ratio, (lo_eff, hi_eff) = self._format_bar(val, lo, hi, i)

            # Format line: label | value | [bar] | min..max
            label_fmt = name[:label_w].ljust(label_w)
            val_fmt = f"{val:+.4f}".rjust(value_w)
            rng_fmt = f"[{lo_eff:+.2f}, {hi_eff:+.2f}]"
            print(f"{label_fmt}  {val_fmt}  {bar}  {rng_fmt}")
