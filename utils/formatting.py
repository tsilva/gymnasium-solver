"""Formatting helpers shared across loggers and reports.

Includes numeric formatting, precision-based value rendering, sort helpers,
and light type checks. Keep dependencies minimal and fail-safe.
"""

from __future__ import annotations

import numbers
from typing import Any


def is_number(x: Any) -> bool:
    """Return True if x is a numeric type (ints, floats, numpy scalars).

    Note: bool is a subclass of int in Python and will return True here.
    """
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
    """Return a path-/project-safe name by replacing separators.

    Replaces forward and back slashes with dashes to avoid unintended
    directory structures in IDs or project names.
    """
    return str(name).replace("/", "-").replace("\\", "-")


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string for a number of seconds.

    Rules:
    - < 1 minute: show seconds with 2 decimals (e.g., "8.98 seconds").
    - < 1 hour: show minutes and seconds (e.g., "12 minutes 5 seconds").
    - < 1 day: show hours, minutes, and seconds (e.g., "1 hour 02 minutes 03 seconds").
    - >= 1 day: show days, hours, minutes, and seconds.
    """
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
    """Format a metric value according to precision in metrics.yaml.

    Accepts either a namespaced key like "val/ep_rew_mean" or a bare metric
    name like "ep_rew_mean". Falls back to precision=2 when unknown.
    Always humanizes magnitudes (k, M, B) for large values.
    """
    try:
        from utils.metrics_config import metrics_config
        bare = str(metric_key_or_bare).rsplit("/", 1)[-1]
        precision_map = metrics_config.metric_precision_dict()
        precision = int(precision_map.get(bare, 2))
    except Exception:
        precision = 2

    try:
        val = float(value)
    except Exception:
        # Best-effort stringification if not numeric
        return str(value)

    return number_to_string(val, precision=precision, humanize=True)
