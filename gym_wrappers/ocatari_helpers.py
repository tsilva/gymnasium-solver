"""Shared helper functions for OCAtari object extraction and normalization."""

import numpy as np
from typing import Tuple


def center(obj) -> Tuple[float, float]:
    """Extract (center_x, center_y) from an OCAtari object.

    Prefers explicit center if present; otherwise derives from (x, y) + (w, h)/2.
    Returns (0.0, 0.0) if obj is None.
    """
    if obj is None:
        return 0.0, 0.0
    if hasattr(obj, "center") and obj.center is not None:
        return float(obj.center[0]), float(obj.center[1])
    x = float(getattr(obj, "x", 0.0))
    y = float(getattr(obj, "y", 0.0))
    w = float(getattr(obj, "w", 0.0))
    h = float(getattr(obj, "h", 0.0))
    return x + 0.5 * w, y + 0.5 * h


def center_x(obj) -> float:
    """Return robust center-X for an OCAtari object.

    Prefer explicit center if present; otherwise derive from left x and width.
    """
    if obj is None:
        return 0.0
    if hasattr(obj, "center") and obj.center is not None:
        return float(obj.center[0])
    w = float(getattr(obj, "w", 0.0))
    x = float(getattr(obj, "x", 0.0))
    return x + 0.5 * w


def center_y(obj) -> float:
    """Return robust center-Y for an OCAtari object.

    Prefer explicit center if present; otherwise derive from top-left y and height.
    """
    if obj is None:
        return 0.0
    if hasattr(obj, "center") and obj.center is not None:
        return float(obj.center[1])
    h = float(getattr(obj, "h", 0.0))
    y = float(getattr(obj, "y", 0.0))
    return y + 0.5 * h


def normalize_velocity(value: float, scale: float) -> float:
    """Map symmetric value in R to [-1, 1] using tanh with scale.

    0 maps to 0.0, extremes saturate to -1/1.
    """
    if scale <= 0:
        return 0.0
    return float(np.tanh(float(value) / float(scale)))
