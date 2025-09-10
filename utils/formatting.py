"""Formatting helpers shared across loggers and reports.

Includes numeric formatting, precision-based value rendering, sort helpers,
and light type checks. Keep dependencies minimal and fail-safe.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple
import numbers


def is_number(x: Any) -> bool:
    """Return True if x is a numeric type (ints, floats, numpy scalars).

    Note: bool is a subclass of int in Python and will return True here.
    """
    return isinstance(x, numbers.Number)


def humanize_num(v, float_fmt=".2f") -> str:
    if isinstance(v, bool): return "1" if v else "0"
    if isinstance(v, int):
        n, s = abs(v), "-"*(v < 0)
        return next((f"{s}{n/d:.2f}{u}" for d,u in ((1_000_000_000,"B"),(1_000_000,"M"),(1_000,"k")) if n >= d), str(v))
    if isinstance(v, float): return f"{v:.2e}" if 0 < abs(v) < 1e-6 else format(v, float_fmt)
    return str(v)


def fmt_plain(v: Any, float_fmt: str = ".2f") -> str:
    """Plain formatter that respects float format but avoids compaction."""
    if isinstance(v, float):
        return format(v, float_fmt)
    return str(v)


def precision_for(full_key: str, precision_map: Dict[str, int]) -> Optional[int]:
    """Return precision for full_key or its bare metric name (after '/')."""
    if full_key in precision_map:
        return int(precision_map[full_key])
    bare = full_key.split("/", 1)[-1]
    if bare in precision_map:
        return int(precision_map[bare])
    return None


def format_value(
    v: Any,
    full_key: str = "",
    *,
    precision_map: Optional[Dict[str, int]] = None,
    compact_numbers: bool = True,
    float_fmt: str = ".2f",
) -> str:
    """Format a metric value considering precision overrides and compaction."""
    if v is None:
        return "—"
    if precision_map is None:
        precision_map = {}
    p = precision_for(full_key, precision_map) if is_number(v) else None
    if p is not None and is_number(v):
        if p == 0:
            return str(int(round(float(v))))
        return f"{float(v):.{p}f}"
    if compact_numbers and is_number(v):
        return humanize_num(v, float_fmt)
    return fmt_plain(v, float_fmt)


def _decimals_from_fmt(fmt: str) -> int:
    try:
        if fmt and fmt.startswith(".") and fmt.endswith("f"):
            return int(fmt[1:-1])
    except Exception:
        pass
    return 2


def format_delta_magnitude(
    delta: numbers.Number,
    full_key: str,
    *,
    precision_map: Optional[Dict[str, int]] = None,
    compact_numbers: bool = True,
    float_fmt: str = ".2f",
) -> str:
    """Format a delta magnitude with adaptive precision/compaction.

    Mirrors the logic used by the console metrics printer.
    """
    precision_map = precision_map or {}
    if isinstance(delta, int):
        return humanize_num(delta, float_fmt) if compact_numbers else fmt_plain(delta, float_fmt)
    if isinstance(delta, float):
        p = precision_for(full_key, precision_map)
        default_decimals = _decimals_from_fmt(float_fmt)
        decimals = int(p) if isinstance(p, int) and p >= 0 else default_decimals
        first = f"{delta:.{decimals}f}"
        if first.strip("0").strip(".") != "":
            try:
                as_float = float(first)
            except ValueError:
                as_float = delta
            if compact_numbers and abs(as_float) >= 1000:
                return humanize_num(as_float, float_fmt)
            return first
        more_decimals = decimals + 2
        second = f"{delta:.{more_decimals}f}"
        if second.strip("0").strip(".") != "":
            return second
        return f"{delta:.2e}"
    return fmt_plain(delta, float_fmt)


# TODO: not reusable
def get_sort_key(namespace: str, subkey: str, key_priority: Iterable[str]) -> Tuple[int, object]:
    """Compute a stable sort key honoring an explicit key priority list.

    Returns (0, priority_index) when full_key in key_priority; otherwise
    (1, subkey.lower()) so prioritized keys appear first in given order.
    """
    full_key = f"{namespace}/{subkey}" if subkey else namespace
    try:
        priority_index = list(key_priority).index(full_key)
        return (0, priority_index)
    except ValueError:
        return (1, subkey.lower())
