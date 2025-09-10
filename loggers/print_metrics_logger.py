from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List, Iterable

import os
import sys
import numbers

from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase  # type: ignore

import torch

class PrintMetricsLogger(LightningLoggerBase):
    """
    Lightning logger that pretty-prints the latest logged metrics as a
    namespaced table, leveraging the same formatting/highlighting rules as
    the previous PrintMetricsCallback but without re-collecting from the
    trainer — it consumes exactly what Lightning dispatches to loggers.

    Intended to be used alongside other loggers (e.g., WandbLogger,
    CsvLightningLogger) so all receive the same metrics payload.
    """

    ANSI_CODES = {
        # Foreground
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
        # Styles
        "bold": "1",
        # Background
        "bg_black": "40",
        "bg_red": "41",
        "bg_green": "42",
        "bg_yellow": "43",
        "bg_blue": "44",
        "bg_magenta": "45",
        "bg_cyan": "46",
        "bg_white": "47",
    }

    def __init__(
        self,
        *,
        metric_precision: Dict[str, int] | None = None,
        metric_delta_rules: Dict[str, Callable] | None = None,
        algorithm_metric_rules: Dict[str, dict] | None = None,
        min_val_width: int = 15,
        key_priority: List[str] | None = None,
    ) -> None:
        from utils.metrics import metrics_config

        # Display config
        self._name = "print"
        self._version: str | int = "0"
        self._experiment = None

        # Rules/config
        self.metric_precision = dict(metric_precision or {})
        self.metric_delta_rules = dict(metric_delta_rules or {})
        self.algorithm_metric_rules = dict(algorithm_metric_rules or {})
        self.min_val_width = int(min_val_width)
        self.key_priority = list(key_priority or [])
        self.previous_metrics: Dict[str, Any] = {}

        # Highlight/bounds from metrics.yaml
        _m = metrics_config
        _hl_cfg = _m.highlight_config()
        _hl_value_bold_for = _hl_cfg.get('value_bold_metrics', set())
        _hl_row_for = _hl_cfg.get('row_metrics', set())
        _hl_row_bg_color = _hl_cfg.get('row_bg_color', 'bg_blue')
        _hl_row_bold = bool(_hl_cfg.get('row_bold', True))
        _metric_bounds = _m.metric_bounds()
        self._metric_bounds = dict(_metric_bounds)

        # -------- Inlined table printer configuration/state --------
        self.float_fmt: str = ".2f"
        self.indent: int = 4
        self.compact_numbers: bool = True
        self.color: bool = bool(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        self.better_when_increasing: Dict[str, bool] = {}
        self.fixed_section_order: Optional[List[str]] = ["train", "val"]
        self.sort_keys_within_section: bool = True
        self.use_ansi_inplace: bool = False
        self.stream = sys.stdout
        self.delta_tol: float = 1e-12
        self.metric_precision_map: Dict[str, int] = dict(self.metric_precision)
        # Highlight settings (bare metric subkeys)
        self.highlight_value_bold_for_set = set(_hl_value_bold_for or {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "epoch"})
        self.highlight_row_for_set = set(_hl_row_for or {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "total_timesteps"})
        self.highlight_row_bg_color: str = _hl_row_bg_color or "bg_blue"
        self.highlight_row_bold: bool = bool(_hl_row_bold)
        # Bounds-based highlighting
        self.metric_bounds_map: Dict[str, Dict[str, float]] = dict(self._metric_bounds or {})
        self.highlight_bounds_bg_color: str = 'bg_yellow'
        # In-memory history for sparklines (per full metric key)
        self._prev: Optional[Dict[str, Any]] = None
        self._last_height: int = 0
        self._history: Dict[str, List[float]] = {}
        self.show_sparklines: bool = True
        self.sparkline_width: int = 32
        self.sparkline_history_cap: int = 512

    # --- Lightning Logger API ---
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    @property
    def version(self) -> str | int:  # pragma: no cover - trivial
        return self._version

    @property
    def experiment(self):  # pragma: no cover - not used
        return self._experiment

    def log_hyperparams(self, params: Any) -> None:  # pragma: no cover - unused
        return None

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        try:
            # Convert values to basic Python scalars for rendering/validation
            simple: Dict[str, Any] = {k: self._to_python_scalar(v) for k, v in dict(metrics).items()}

            # Validate deltas and algorithm-specific rules using the latest snapshot
            self._validate_metric_deltas(simple)
            self._check_algorithm_metric_rules(simple)

            # Render the metrics table
            self._render_table(simple)

            # Track across calls
            self.previous_metrics.update(simple)
        except Exception:
            # Never block training on printing issues
            pass

    def finalize(self, status: str) -> None:  # pragma: no cover - best-effort noop
        return None

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:  # pragma: no cover - unused
        return None

    # --- Helpers ---
    def _to_python_scalar(self, x: Any) -> Any:
        try:
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.detach().item()
                return x.detach().float().mean().item()
            if hasattr(x, "item") and callable(getattr(x, "item")):
                return x.item()
            return x
        except Exception:
            return x

    def _is_number(self, x: Any) -> bool:
        return isinstance(x, numbers.Number)

    def _validate_metric_deltas(self, current_metrics: Dict[str, Any]) -> None:
        for metric_name, rule_lambda in self.metric_delta_rules.items():
            if metric_name in current_metrics and metric_name in self.previous_metrics:
                curr = current_metrics[metric_name]
                prev = self.previous_metrics[metric_name]
                if not (self._is_number(curr) and self._is_number(prev)):
                    continue
                try:
                    ok = bool(rule_lambda(prev, curr))
                except Exception:
                    ok = True
                if ok:
                    continue
                # Emit a clear violation message but do not raise
                print(
                    f"⚠️  Metric delta rule violation for '{metric_name}': previous={prev}, current={curr}."
                )

    def _check_algorithm_metric_rules(self, current_metrics: Dict[str, Any]) -> None:
        for metric_name, rule_config in self.algorithm_metric_rules.items():
            if metric_name not in current_metrics:
                continue
            curr = current_metrics[metric_name]
            if not self._is_number(curr):
                continue
            try:
                check_func = rule_config.get('check')
                message_template = rule_config.get('message', f"Algorithm metric rule violated for '{metric_name}'")
                level = rule_config.get('level', 'warning')
                if check_func is None:
                    continue
                satisfied = False
                if callable(check_func):
                    try:
                        if getattr(check_func, "__code__", None) and check_func.__code__.co_argcount == 1:
                            satisfied = bool(check_func(curr))
                        elif metric_name in self.previous_metrics:
                            prev = self.previous_metrics.get(metric_name)
                            if self._is_number(prev):
                                satisfied = bool(check_func(prev, curr))
                            else:
                                satisfied = True
                        else:
                            satisfied = True
                    except Exception:
                        satisfied = True
                if not satisfied:
                    prev = self.previous_metrics.get(metric_name, 'N/A')
                    msg = message_template.format(metric_name=metric_name, current_value=curr, previous_value=prev)
                    if level == 'error':
                        print(f"🚨 ALGORITHM ERROR: {msg}")
                    else:
                        print(f"⚠️  ALGORITHM WARNING: {msg}")
            except Exception:
                # Be resilient on logging
                pass

    # ------------- Rendering helpers (merged from NamespaceTablePrinter) -------------
    @classmethod
    def _ansi(cls, color: Optional[str], s: str, enable: bool) -> str:
        if not enable or not color:
            return s
        code = cls.ANSI_CODES.get(color)
        if not code:
            return s
        return f"\x1b[{code}m{s}\x1b[0m"

    @classmethod
    def _apply_row_background(cls, text: str, bg_color: str, enable: bool) -> str:
        if not enable or not bg_color:
            return text
        code = cls.ANSI_CODES.get(bg_color)
        if not code:
            return text
        start = f"\x1b[{code}m"
        body = text.replace("\x1b[0m", f"\x1b[0m{start}")
        return f"{start}{body}\x1b[0m"

    @staticmethod
    def _humanize_num(v: numbers.Number, float_fmt: str = ".2f") -> str:
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, int):
            n = abs(v)
            sign = "-" if v < 0 else ""
            if n >= 1_000_000_000:
                return f"{sign}{n/1_000_000_000:.2f}B"
            if n >= 1_000_000:
                return f"{sign}{n/1_000_000:.2f}M"
            if n >= 1_000:
                return f"{sign}{n/1_000:.2f}k"
            return str(v)
        if isinstance(v, float):
            if 0 < abs(v) < 1e-6:
                return f"{v:.2e}"
            return format(v, float_fmt)
        return str(v)

    @staticmethod
    def _fmt_plain(v: Any, float_fmt: str = ".2f") -> str:
        if isinstance(v, float):
            return format(v, float_fmt)
        return str(v)

    def _group_by_namespace(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if "/" in k:
                ns, sub = k.split("/", 1)
            else:
                ns, sub = k, ""
            grouped.setdefault(ns, {})[sub] = v
        return grouped

    def _precision_for(self, full_key: str) -> Optional[int]:
        if full_key in self.metric_precision_map:
            return int(self.metric_precision_map[full_key])
        # Fallback to bare metric name (post-slash)
        bare = full_key.split("/", 1)[-1]
        if bare in self.metric_precision_map:
            return int(self.metric_precision_map[bare])
        return None

    def _format_value(self, v: Any, full_key: str = "") -> str:
        if v is None:
            return "—"
        precision = self._precision_for(full_key) if self._is_number(v) else None
        if precision is not None and self._is_number(v):
            if precision == 0:
                try:
                    return str(int(round(float(v))))
                except Exception:
                    return str(v)
            return f"{float(v):.{precision}f}"
        if self.compact_numbers and self._is_number(v):
            return self._humanize_num(v, self.float_fmt)
        return self._fmt_plain(v, self.float_fmt)

    def _delta_for_key(self, ns: str, sub: str, v: Any):
        if self._prev is None:
            return ("", None)
        full_key = f"{ns}/{sub}" if sub else ns
        if full_key not in self._prev:
            return ("", None)
        prev_v = self._prev[full_key]
        if not (self._is_number(v) and self._is_number(prev_v)):
            return ("", None)
        delta = float(v) - float(prev_v)
        if abs(delta) <= self.delta_tol:
            return ("→0", "gray")
        arrow = "↑" if delta > 0 else "↓"
        if full_key in self.better_when_increasing:
            inc_better = self.better_when_increasing[full_key]
            improved = (delta > 0) if inc_better else (delta < 0)
            color = "green" if improved else "red"
        else:
            color = "green" if delta > 0 else "red"
        mag = self._format_delta_magnitude(abs(delta), full_key)
        return (f"{arrow}{mag}", color)

    def _format_delta_magnitude(self, delta: numbers.Number, full_key: str) -> str:
        if isinstance(delta, int):
            return self._humanize_num(delta, self.float_fmt) if self.compact_numbers else self._fmt_plain(delta, self.float_fmt)
        try:
            import numpy as _np
            if isinstance(delta, (_np.generic,)):
                delta = delta.item()
        except Exception:
            pass
        if isinstance(delta, float):
            precision = self._precision_for(full_key)

            def _decimals_from_fmt(fmt: str) -> int:
                try:
                    if fmt and fmt.startswith(".") and fmt.endswith("f"):
                        return int(fmt[1:-1])
                except Exception:
                    pass
                return 2

            default_decimals = _decimals_from_fmt(self.float_fmt)
            decimals = int(precision) if isinstance(precision, int) and precision >= 0 else default_decimals
            first = f"{delta:.{decimals}f}"
            if first.strip("0").strip(".") != "":
                try:
                    as_float = float(first)
                except ValueError:
                    as_float = delta
                if self.compact_numbers and abs(as_float) >= 1000:
                    return self._humanize_num(as_float, self.float_fmt)
                return first
            more_decimals = decimals + 2
            second = f"{delta:.{more_decimals}f}"
            if second.strip("0").strip(".") != "":
                return second
            return f"{delta:.2e}"
        return self._fmt_plain(delta, self.float_fmt)

    def _get_sort_key(self, namespace: str, subkey: str) -> tuple:
        full_key = f"{namespace}/{subkey}" if subkey else namespace
        try:
            priority_index = self.key_priority.index(full_key)
            return (0, priority_index)
        except ValueError:
            return (1, subkey.lower())

    @staticmethod
    def _strip_ansi(s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            if s[i] == "\x1b":
                while i < len(s) and s[i] != "m":
                    i += 1
                if i < len(s):
                    i += 1
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

    def _downsample(self, seq: List[float], target: int) -> List[float]:
        if len(seq) <= target:
            return list(seq)
        step = len(seq) / float(target)
        return [seq[int(i * step)] for i in range(target)]

    def _spark_for_key(self, full_key: str, width: int) -> str:
        values = self._history.get(full_key)
        if not values or len(values) < 2 or width <= 0:
            return ""
        data = self._downsample(values, max(1, width))
        vmin = min(data)
        vmax = max(data)
        if vmax == vmin:
            return "─" * min(width, len(data))
        blocks = "▁▂▃▄▅▆▇█"
        rng = (vmax - vmin) or 1.0
        out_chars: List[str] = []
        for v in data:
            idx = int((v - vmin) / rng * (len(blocks) - 1))
            idx = max(0, min(idx, len(blocks) - 1))
            out_chars.append(blocks[idx])
        return "".join(out_chars)

    def _update_history(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if not self._is_number(v):
                continue
            val = float(v)
            hist = self._history.setdefault(k, [])
            hist.append(val)
            if len(hist) > self.sparkline_history_cap:
                self._history[k] = hist[-self.sparkline_history_cap :]

    def _render_lines(self, lines: List[str]) -> None:
        text = "\n".join(lines)
        if self.use_ansi_inplace and self._last_height > 0:
            self.stream.write(f"\x1b[{self._last_height}F")
            self.stream.write("\x1b[0J")
            self.stream.write(text + "\n")
            self.stream.flush()
        else:
            if not self.use_ansi_inplace:
                os.system('cls' if os.name == 'nt' else 'clear')
            print(text, file=self.stream)

    def _render_table(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        self._update_history(data)
        grouped = self._group_by_namespace(data)
        ns_names = list(grouped.keys())
        if self.fixed_section_order:
            pref = [ns for ns in self.fixed_section_order if ns in grouped]
            rest = sorted([ns for ns in ns_names if ns not in self.fixed_section_order])
            ns_order = pref + rest
        else:
            ns_order = sorted(ns_names)
        for ns in ns_order:
            if self.sort_keys_within_section:
                grouped[ns] = dict(sorted(grouped[ns].items(), key=lambda kv: self._get_sort_key(ns, kv[0])))
        formatted: Dict[str, Dict[str, str]] = {}
        val_candidates: List[str] = []
        key_candidates: List[str] = [ns + "/" for ns in ns_order]
        for ns in ns_order:
            subdict = grouped[ns]
            f_sub: Dict[str, str] = {}
            for sub, v in subdict.items():
                key_candidates.append(sub)
                full_key = f"{ns}/{sub}" if sub else ns
                val_str = self._format_value(v, full_key)
                try:
                    if sub in self.highlight_value_bold_for_set:
                        val_str = self._ansi("bold", val_str, self.color)
                except Exception:
                    pass
                delta_str, color_name = self._delta_for_key(ns, sub, v)
                if delta_str:
                    delta_disp = self._ansi(color_name, delta_str, self.color)
                    val_disp = f"{val_str} {delta_disp}"
                else:
                    val_disp = val_str
                try:
                    if self.show_sparklines and self._is_number(v):
                        chart = self._spark_for_key(full_key, self.sparkline_width)
                        if chart:
                            val_disp = f"{val_disp}  {chart}"
                except Exception:
                    pass
                f_sub[sub] = val_disp
                val_candidates.append(self._strip_ansi(val_disp))
            formatted[ns] = f_sub
        indent = self.indent
        key_width = max((len(k) for k in key_candidates), default=0)
        val_width = max((len(v) for v in val_candidates), default=0)
        val_width = max(val_width, self.min_val_width)
        border_len = 2 + (indent + key_width) + 3 + val_width + 2
        border = "-" * border_len
        lines: List[str] = []
        lines.append(border)
        for ns in ns_order:
            if not formatted.get(ns):
                continue
            header = ns + "/"
            header_line = f"| {header:<{indent + key_width}} | {'':>{val_width}} |"
            lines.append(self._ansi("bold", header_line, self.color))
            for sub, val in formatted[ns].items():
                val_display_len = len(self._strip_ansi(val))
                val_padding = val_width - val_display_len
                val_padded = (" " * val_padding + val) if val_padding > 0 else val
                key_cell = f"{sub:<{key_width}}"
                highlight = False
                row_bg_color = None
                try:
                    full_key = f"{ns}/{sub}" if sub else ns
                    # Priority 1: bounds-based highlight (yellow); allow bare-name lookup
                    bounds = self.metric_bounds_map.get(full_key) or self.metric_bounds_map.get(sub)
                    if bounds:
                        raw_val = grouped.get(ns, {}).get(sub)
                        if self._is_number(raw_val):
                            vnum = float(raw_val)
                            below = ("min" in bounds) and (vnum < float(bounds["min"]))
                            above = ("max" in bounds) and (vnum > float(bounds["max"]))
                            if below or above:
                                highlight = True
                                row_bg_color = self.highlight_bounds_bg_color
                    # Priority 2: configured row highlight
                    if not highlight and sub in self.highlight_row_for_set:
                        if self.highlight_row_bold:
                            key_cell = self._ansi("bold", key_cell, self.color)
                        highlight = True
                        row_bg_color = self.highlight_row_bg_color
                except Exception:
                    pass
                row = f"| {' ' * indent}{key_cell} | {val_padded} |"
                if highlight:
                    row = self._apply_row_background(row, row_bg_color or self.highlight_row_bg_color, self.color)
                lines.append(row)
        lines.append(border)
        self._render_lines(lines)
        self._prev = dict(data)
        self._last_height = len(lines)

    # ------------- One-off rendering entrypoint for external callers -------------
    @classmethod
    def render_namespaced_dict(
        cls,
        data: Dict[str, Any],
        *,
        inplace: bool = False,
        float_fmt: str = ".2f",
        compact_numbers: bool = True,
        color: bool = True,
        metric_precision: Optional[Dict[str, int]] = None,
        min_val_width: int = 15,
        key_priority: Optional[List[str]] = None,
        highlight_value_bold_for: Optional[Iterable[str]] = None,
        highlight_row_for: Optional[Iterable[str]] = None,
        highlight_row_bg_color: str = "bg_blue",
        highlight_row_bold: bool = True,
        metric_bounds: Optional[Dict[str, Dict[str, float]]] = None,
        highlight_bounds_bg_color: str = "bg_yellow",
    ) -> None:
        # Create a temporary instance for stateless, one-off printing
        inst = cls(
            metric_precision=metric_precision or {},
            metric_delta_rules={},
            algorithm_metric_rules={},
            min_val_width=min_val_width,
            key_priority=key_priority or [],
        )
        inst.float_fmt = float_fmt
        inst.compact_numbers = compact_numbers
        inst.use_ansi_inplace = bool(inplace and sys.stdout.isatty())
        inst.color = bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        # Override highlights if provided
        if highlight_value_bold_for is not None:
            inst.highlight_value_bold_for_set = set(highlight_value_bold_for)
        if highlight_row_for is not None:
            inst.highlight_row_for_set = set(highlight_row_for)
        if highlight_row_bg_color:
            inst.highlight_row_bg_color = highlight_row_bg_color
        inst.highlight_row_bold = bool(highlight_row_bold)
        # Bounds
        if metric_bounds is not None:
            inst.metric_bounds_map = dict(metric_bounds)
        if highlight_bounds_bg_color:
            inst.highlight_bounds_bg_color = highlight_bounds_bg_color
        # Render once (no history/deltas across calls)
        inst._render_table(data)
