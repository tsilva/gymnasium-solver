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
        for magnitude, unit in magnitudes:
            if value < magnitude: continue
            value /= magnitude
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
