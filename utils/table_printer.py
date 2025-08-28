"""Namespace-aware metrics table printer utilities."""

from __future__ import annotations

import os
import sys
import numbers
from typing import Any, Dict, Iterable, List, Optional


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Number)


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


def _fmt_plain(v: Any, float_fmt: str = ".2f") -> str:
    if isinstance(v, float):
        return format(v, float_fmt)
    return str(v)


def _ansi(color: Optional[str], s: str, enable: bool) -> str:
    if not enable or not color:
        return s
    codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
        "bold": "1",
        # Background colors
        "bg_black": "40",
        "bg_red": "41",
        "bg_green": "42",
        "bg_yellow": "43",
        "bg_blue": "44",
        "bg_magenta": "45",
        "bg_cyan": "46",
        "bg_white": "47",
    }
    if color == "bold":
        return f"\x1b[{codes['bold']}m{s}\x1b[0m"
    return f"\x1b[{codes[color]}m{s}\x1b[0m"


def _apply_row_background(text: str, bg_color: str, enable: bool) -> str:
    """
    Apply a background color to an entire ANSI-formatted line, preserving
    inner foreground color/bold segments. Ensures that any embedded resets
    (\x1b[0m) re-apply the background so the highlight spans the full row.
    """
    if not enable or not bg_color:
        return text
    bg_map = {
        "bg_black": "40",
        "bg_red": "41",
        "bg_green": "42",
        "bg_yellow": "43",
        "bg_blue": "44",
        "bg_magenta": "45",
        "bg_cyan": "46",
        "bg_white": "47",
    }
    code = bg_map.get(bg_color)
    if not code:
        return text
    start = f"\x1b[{code}m"
    # Re-apply background after any full reset encountered in the row
    body = text.replace("\x1b[0m", f"\x1b[0m{start}")
    return f"{start}{body}\x1b[0m"


class NamespaceTablePrinter:
    def __init__(
        self,
        *,
        float_fmt: str = ".2f",
        indent: int = 4,
        compact_numbers: bool = True,
        color: bool = True,
        better_when_increasing: Optional[Dict[str, bool]] = None,
        fixed_section_order: Optional[Iterable[str]] = ("train", "eval"),
        sort_keys_within_section: bool = True,
        use_ansi_inplace: bool = True,
        stream=None,
        delta_tol: float = 1e-12,
        metric_precision: Optional[Dict[str, int]] = None,
        min_val_width: int = 15,
        key_priority: Optional[List[str]] = None,
    ):
        self.float_fmt = float_fmt
        self.indent = indent
        self.compact_numbers = compact_numbers
        self.color = bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        self.better_when_increasing = dict(better_when_increasing or {})
        self.fixed_section_order = list(fixed_section_order) if fixed_section_order else None
        self.sort_keys_within_section = sort_keys_within_section
        self.use_ansi_inplace = use_ansi_inplace and sys.stdout.isatty()
        self.stream = stream or sys.stdout
        self.delta_tol = delta_tol
        self.metric_precision = dict(metric_precision or {})
        self.min_val_width = min_val_width
        self.key_priority = key_priority or []

        self._prev: Optional[Dict[str, Any]] = None
        self._last_height: int = 0

    def update(self, data: Dict[str, Any]) -> None:
        if not data:
            return

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
                grouped[ns] = dict(
                    sorted(grouped[ns].items(), key=lambda kv: self._get_sort_key(ns, kv[0]))
                )

        formatted = {}
        val_candidates = []
        key_candidates = [ns + "/" for ns in ns_order]
        for ns in ns_order:
            subdict = grouped[ns]
            f_sub = {}
            for sub, v in subdict.items():
                key_candidates.append(sub)
                full_key = f"{ns}/{sub}" if sub else ns
                val_str = self._format_value(v, full_key)
                # Emphasize key episode reward metrics for readability
                try:
                    if sub in {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "epoch"}:
                        val_str = _ansi("bold", val_str, self.color)
                except Exception:
                    pass
                delta_str, color_name = self._delta_for_key(ns, sub, v)
                if delta_str:
                    delta_disp = _ansi(color_name, delta_str, self.color)
                    val_disp = f"{val_str} {delta_disp}"
                else:
                    val_disp = val_str
                f_sub[sub] = val_disp
                val_candidates.append(self._strip_ansi(val_disp))
            formatted[ns] = f_sub

        indent = self.indent
        key_width = max(len(k) for k in key_candidates) if key_candidates else 0
        val_width = max(len(v) for v in val_candidates) if val_candidates else 0
        val_width = max(val_width, self.min_val_width)

        border_len = 2 + (indent + key_width) + 3 + val_width + 2
        border = "-" * border_len

        lines = []
        lines.append(border)
        for ns in ns_order:
            if not formatted.get(ns):
                continue
            header = ns + "/"
            header_line = f"| {header:<{indent + key_width}} | {'':>{val_width}} |"
            lines.append(_ansi("bold", header_line, self.color))
            for sub, val in formatted[ns].items():
                val_display_len = len(self._strip_ansi(val))
                val_padding = val_width - val_display_len
                val_padded = " " * val_padding + val if val_padding > 0 else val
                # Format key cell with padding first, then apply ANSI bold if highlighted
                key_cell = f"{sub:<{key_width}}"
                highlight = False
                try:
                    if sub in {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "total_timesteps"}:
                        key_cell = _ansi("bold", key_cell, self.color)
                        highlight = True
                except Exception:
                    pass
                row = f"| {' ' * indent}{key_cell} | {val_padded} |"
                if highlight:
                    # Apply a subtle background to the entire row for visibility
                    row = _apply_row_background(row, "bg_blue", self.color)
                lines.append(row)
        lines.append(border)

        self._render_lines(lines)
        self._prev = dict(data)
        self._last_height = len(lines)

    def _render_lines(self, lines):
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

    def _group_by_namespace(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if "/" in k:
                ns, sub = k.split("/", 1)
            else:
                ns, sub = k, ""
            grouped.setdefault(ns, {})[sub] = v
        return grouped

    def _format_value(self, v: Any, full_key: str = "") -> str:
        if v is None:
            return "—"
        if full_key in self.metric_precision:
            precision = self.metric_precision[full_key]
            if _is_number(v):
                if precision == 0:
                    return str(int(round(v)))
                else:
                    return f"{v:.{precision}f}"
        if self.compact_numbers and _is_number(v):
            return _humanize_num(v, self.float_fmt)
        return _fmt_plain(v, self.float_fmt)

    def _delta_for_key(self, ns: str, sub: str, v: Any):
        if self._prev is None:
            return ("", None)
        full_key = f"{ns}/{sub}" if sub else ns
        if full_key not in self._prev:
            return ("", None)
        prev_v = self._prev[full_key]
        if not (_is_number(v) and _is_number(prev_v)):
            return ("", None)
        delta = v - prev_v
        if abs(delta) <= self.delta_tol:
            return ("→0", "gray")
        color = None
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
            return _humanize_num(delta, self.float_fmt) if self.compact_numbers else _fmt_plain(delta, self.float_fmt)
        try:
            import numpy as _np
            if isinstance(delta, (_np.generic,)):
                delta = delta.item()
        except Exception:
            pass
        if isinstance(delta, float):
            precision = self.metric_precision.get(full_key)

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
                    return _humanize_num(as_float, self.float_fmt)
                return first
            more_decimals = decimals + 2
            second = f"{delta:.{more_decimals}f}"
            if second.strip("0").strip(".") != "":
                return second
            return f"{delta:.2e}"
        return _fmt_plain(delta, self.float_fmt)

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


_default_printer = NamespaceTablePrinter()


def print_namespaced_dict(
    data: Dict[str, Any],
    *,
    inplace: bool = False,
    float_fmt: str = ".2f",
    compact_numbers: bool = True,
    color: bool = True,
    metric_precision: Optional[Dict[str, int]] = None,
    min_val_width: int = 15,
    key_priority: Optional[List[str]] = None,
):
    global _default_printer
    if (
        _default_printer.float_fmt != float_fmt
        or _default_printer.compact_numbers != compact_numbers
        or (_default_printer.use_ansi_inplace != bool(inplace and sys.stdout.isatty()))
        or _default_printer.color != bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        or _default_printer.metric_precision != (metric_precision or {})
        or _default_printer.min_val_width != min_val_width
        or _default_printer.key_priority != (key_priority or [])
    ):
        _default_printer = NamespaceTablePrinter(
            float_fmt=float_fmt,
            compact_numbers=compact_numbers,
            use_ansi_inplace=bool(inplace and sys.stdout.isatty()),
            color=bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None),
            metric_precision=metric_precision,
            min_val_width=min_val_width,
            key_priority=key_priority,
        )
    _default_printer.update(data)
