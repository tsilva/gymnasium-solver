from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
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
        # Reward range from spec (lo, hi) when available
        self._reward_range: Tuple[Optional[float], Optional[float]] = (None, None)
        # Action metadata (from spec or action_space)
        self._action_discrete_n: Optional[int] = None
        self._action_labels: Optional[Dict[int, str]] = None
        self._last_actions: Optional[np.ndarray] = None
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
        self._sum_ep_returns = 0.0
        # Print initial observation if enabled
        if self._enable:
            self._print_obs(obs, rewards=None, dones=None)
        return obs

    def step_async(self, actions):
        # Cache last actions for rendering (best-effort; shapes vary by env)
        try:
            self._last_actions = np.asarray(actions)
        except Exception:
            self._last_actions = None
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
                # Update running sum for mean across played episodes
                try:
                    self._sum_ep_returns += float(self._last_ep_return)
                except Exception:
                    pass
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
                spec_obj = self.venv.get_spec()
                if hasattr(spec_obj, "as_dict"):
                    spec = spec_obj.as_dict()
                elif isinstance(spec_obj, Mapping):
                    spec = dict(spec_obj)
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

        # Parse reward range from spec when available
        try:
            rewards = spec.get("rewards") if isinstance(spec, dict) else None
            if isinstance(rewards, dict):
                rr = rewards.get("range")
                if isinstance(rr, Sequence) and len(rr) == 2:
                    lo = _parse_inf(rr[0])
                    hi = _parse_inf(rr[1])
                    self._reward_range = (lo, hi)
        except Exception:
            pass

        # Parse action space metadata (discrete count, optional labels)
        try:
            act = spec.get("action_space") if isinstance(spec, dict) else None
            if isinstance(act, dict):
                if "discrete" in act and isinstance(act.get("discrete"), (int, float)):
                    self._action_discrete_n = int(act.get("discrete"))
                labels_map = act.get("labels")
                if isinstance(labels_map, dict):
                    parsed: Dict[int, str] = {}
                    for k, v in labels_map.items():
                        try:
                            parsed[int(k)] = str(v)
                        except Exception:
                            continue
                    if parsed:
                        self._action_labels = parsed
        except Exception:
            pass

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

    def _format_bar_scalar(self, value: float, lo: Optional[float], hi: Optional[float], *, width: Optional[int] = None) -> Tuple[str, float, Tuple[float, float]]:
        """Format a bar for a single scalar value using provided bounds only.

        Unlike _format_bar, this does not update dynamic observation ranges and is
        suitable for non-observation quantities such as rewards.
        """
        lo_eff = float(lo) if (lo is not None and math.isfinite(float(lo))) else None
        hi_eff = float(hi) if (hi is not None and math.isfinite(float(hi))) else None
        if lo_eff is None or hi_eff is None or hi_eff == lo_eff:
            lo_eff, hi_eff = -1.0, 1.0
        bw = self._bar_width if width is None else max(1, int(width))
        ratio = 0.0 if hi_eff == lo_eff else (float(value) - lo_eff) / (hi_eff - lo_eff)
        ratio = max(0.0, min(1.0, ratio))
        filled = int(round(ratio * bw))
        empty = bw - filled
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

        # Optional status line (only done flag now; reward shown below as a bar)
        status_parts: List[str] = []
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

        # Build header with episode info, last reward, and running mean reward
        curr_ep_idx = self._ep_count + 1  # 1-based index for current episode
        if self._target_episodes is not None and self._target_episodes > 0:
            ep_prog = f"Ep {curr_ep_idx}/{self._target_episodes}"
        else:
            ep_prog = f"Ep {curr_ep_idx}"
        # Current episode progress (return and length)
        cur_ep = f"cur_ep_r={self._current_ep_return:+.3f}"
        cur_len = f"cur_len={int(self._current_ep_len)}"

        last_ep = (
            f"last_ep_r={self._last_ep_return:+.3f}"
            if isinstance(self._last_ep_return, (int, float))
            else "last_ep_r=--"
        )
        mean_ep = (
            f"ep_rew_mean={(self._sum_ep_returns / self._ep_count):+.3f}"
            if isinstance(self._ep_count, int) and self._ep_count > 0
            else "ep_rew_mean=--"
        )
        header_main = "  ".join(
            x for x in [env_id or "", ep_prog, cur_len, cur_ep, last_ep, mean_ep] if x
        )
        print(header_main)
        if status:
            print(status)

        # Episode timestep progress bar, if time limit is known
        if self._time_limit is None:
            try:
                if hasattr(self.venv, "get_time_limit"):
                    tl = self.venv.get_time_limit()
                    if isinstance(tl, (int, float)) and tl and int(tl) > 0:
                        self._time_limit = int(tl)
            except Exception:
                self._time_limit = None
        if isinstance(self._time_limit, int) and self._time_limit > 0:
            steps = int(self._current_ep_len)
            total = int(self._time_limit)
            t_bar, _t_ratio, (_t_lo, _t_hi) = self._format_bar_scalar(steps, 0, total, width=self._bar_width)
            label_fmt = "timestep"[:label_w].ljust(label_w)
            val_fmt = f"{steps}/{total}".rjust(value_w)
            t_rng_fmt = f"[0, {total}]"
            print(f"{label_fmt}  {val_fmt}  {t_bar}  {t_rng_fmt}")

        # Reward bar (scaled using spec reward range when available)
        reward_val: Optional[float] = None
        try:
            if isinstance(rewards, np.ndarray) and rewards.size > 0:
                reward_val = float(rewards[self._env_index])
        except Exception:
            reward_val = None
        if reward_val is not None:
            r_lo, r_hi = self._reward_range
            r_bar, _r_ratio, (r_lo_eff, r_hi_eff) = self._format_bar_scalar(reward_val, r_lo, r_hi, width=self._bar_width)
            label_fmt = "reward"[:label_w].ljust(label_w)
            val_fmt = f"{reward_val:+.4f}".rjust(value_w)
            r_rng_fmt = f"[{r_lo_eff:+.2f}, {r_hi_eff:+.2f}]"
            print(f"{label_fmt}  {val_fmt}  {r_bar}  {r_rng_fmt}")

        # Action bar (discrete only): scale 0..N-1 with last action value
        action_idx: Optional[int] = None
        try:
            if self._last_actions is not None:
                arr = np.asarray(self._last_actions)
                if arr.ndim == 0:
                    action_idx = int(arr.item())
                elif arr.ndim == 1 and arr.size > self._env_index:
                    action_idx = int(arr[self._env_index])
                elif arr.ndim >= 2 and arr.shape[0] > self._env_index:
                    # Handle shapes like (n_envs, 1)
                    action_idx = int(np.asarray(arr[self._env_index]).flatten()[0])
        except Exception:
            action_idx = None
        # Determine discrete action count
        act_n: Optional[int] = None
        if isinstance(self.action_space, spaces.Discrete):
            act_n = int(self.action_space.n)
        elif isinstance(self._action_discrete_n, int) and self._action_discrete_n > 0:
            act_n = int(self._action_discrete_n)
        if (action_idx is not None) and (act_n is not None) and act_n > 0:
            a_lo, a_hi = 0, act_n - 1
            a_bar, _a_ratio, (_a_lo, _a_hi) = self._format_bar_scalar(float(action_idx), float(a_lo), float(a_hi), width=self._bar_width)
            label_fmt = "action"[:label_w].ljust(label_w)
            val_fmt = f"{action_idx}".rjust(value_w)
            a_rng_fmt = f"[{a_lo}, {a_hi}]"
            print(f"{label_fmt}  {val_fmt}  {a_bar}  {a_rng_fmt}")

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
