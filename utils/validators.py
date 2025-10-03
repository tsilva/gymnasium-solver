"""Reusable validation helpers."""

from typing import Any


def ensure_positive(value: Any, field_name: str, *, allow_none: bool = True) -> None:
    """Ensure that ``value`` is positive when provided."""
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{field_name} must be a positive number when set.")
    if not (value > 0):  # type: ignore[operator]
        message = "float/int" if allow_none else "number"
        raise ValueError(f"{field_name} must be a positive {message} when set.")


def ensure_non_negative(value: Any, field_name: str, *, allow_none: bool = True) -> None:
    """Ensure that ``value`` is non-negative when provided."""
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{field_name} must be a non-negative number when set.")
    if not (value >= 0):  # type: ignore[operator]
        raise ValueError(f"{field_name} must be a non-negative number when set.")


def ensure_in_range(
    value: Any,
    field_name: str,
    min_val: float,
    max_val: float,
    *,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> None:
    """Ensure that ``value`` lies within the requested range when provided."""
    if value is None:
        return

    min_ok = (value >= min_val) if inclusive_min else (value > min_val)
    max_ok = (value <= max_val) if inclusive_max else (value < max_val)
    if min_ok and max_ok:
        return

    min_bracket = "[" if inclusive_min else "("
    max_bracket = "]" if inclusive_max else ")"
    raise ValueError(
        f"{field_name} must be in {min_bracket}{min_val}, {max_val}{max_bracket} when set."
    )
