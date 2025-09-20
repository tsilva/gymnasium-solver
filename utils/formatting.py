"""Formatting helpers for numbers, durations, and safe names."""

from __future__ import annotations

import numbers
from typing import Any


def is_number(x: Any) -> bool:
    """Return True when x is a numeric type (ints, floats, numpy scalars)."""
    return isinstance(x, numbers.Number)


def number_to_string(
    value: Any,
    *,
    precision: int = 2,
    humanize: bool = True,
) -> str:
    """Format a metric value considering precision overrides and compaction."""
    if value is None: return "—"

    unit = ""
    magnitudes = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")]
    if humanize:
        for _magnitude, _unit in magnitudes:
            if value < _magnitude: continue
            value /= _magnitude
            unit = _unit
            precision = 2
            break
        
    if precision == 0: value_str = str(int(round(value))) + unit
    else: value_str = f"{float(value):.{precision}f}" + unit

    return value_str


def sanitize_name(name: str) -> str:
    """Return a path-/project-safe name by replacing path separators."""
    return str(name).replace("/", "-").replace("\\", "-")


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string for seconds (minutes/hours/days as needed)."""
    try:
        s = max(0.0, float(seconds))
    except Exception:
        return "0 seconds"

    # Sub-minute: preserve fractional seconds with two decimals
    if s < 60.0:
        return f"{s:.2f} seconds"

    # For minute+ ranges, round to whole seconds and decompose
    total = int(round(s))
    days = total // 86_400
    rem = total % 86_400
    hours = rem // 3_600
    rem %= 3_600
    minutes = rem // 60
    secs = rem % 60

    # Build per-range strings
    if total < 3_600:
        # minutes + seconds
        parts = [
            f"{minutes} {_plural(minutes, 'minute', 'minutes')}",
            f"{secs} {_plural(secs, 'second', 'seconds')}",
        ]
        return " ".join(parts)

    if total < 86_400:
        parts = [
            f"{hours} {_plural(hours, 'hour', 'hours')}",
            f"{minutes} {_plural(minutes, 'minute', 'minutes')}",
            f"{secs} {_plural(secs, 'second', 'seconds')}",
        ]
        return " ".join(parts)

    # 1 day or more
    parts = [
        f"{days} {_plural(days, 'day', 'days')}",
        f"{hours} {_plural(hours, 'hour', 'hours')}",
        f"{minutes} {_plural(minutes, 'minute', 'minutes')}",
        f"{secs} {_plural(secs, 'second', 'seconds')}",
    ]
    return " ".join(parts)


def format_metric_value(metric_key_or_bare: str, value: Any) -> str:
    """Format a metric value using metrics.yaml precision (humanizes large magnitudes)."""
    try:
        from utils.metrics_config import metrics_config
        bare = str(metric_key_or_bare).rsplit("/", 1)[-1]
        precision_map = metrics_config._build_metric_precision_dict()
        precision = int(precision_map.get(bare, 2))
    except Exception:
        precision = 2

    try:
        val = float(value)
    except Exception:
        # Best-effort stringification if not numeric
        return str(value)

    return number_to_string(val, precision=precision, humanize=True)
