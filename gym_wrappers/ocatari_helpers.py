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


def normalize_linear(value: float, lo: float, hi: float) -> float:
    """Normalize a value to [-1, 1] over the range [lo, hi].

    Maps lo -> -1.0, hi -> 1.0, with linear interpolation.
    """
    assert hi > lo, f"Invalid range [{lo}, {hi}]"
    zero_one = (value - lo) / (hi - lo)
    return 2.0 * zero_one - 1.0


def normalize_position(
    center: float,
    min_bound: float,
    max_bound: float,
    object_size: float,
    margin: float = 0.0
) -> float:
    """Normalize object center position to [-1, 1] accounting for object size and margin.

    Computes bounds as:
        lo = min_bound + object_size/2 - margin
        hi = max_bound - object_size/2 + margin

    Then maps center linearly from [lo, hi] to [-1, 1].
    """
    lo = min_bound + 0.5 * object_size - margin
    hi = max_bound - 0.5 * object_size + margin
    return normalize_linear(center, lo, hi)


def index_objects_by_category(objects, skip_hud: bool = True, skip_no_object: bool = True, assert_unique: bool = True):
    """Index OCAtari objects by category, skipping HUD and NoObject placeholders.

    Args:
        objects: Iterable of OCAtari objects
        skip_hud: Skip objects with hud=True attribute
        skip_no_object: Skip objects with category="NoObject"
        assert_unique: Assert that each category appears at most once

    Returns:
        Dict mapping category string to object
    """
    objects_map = {}
    for obj in objects:
        # Skip HUD objects
        if skip_hud and getattr(obj, "hud", False):
            continue

        category = getattr(obj, "category", None)

        # Skip NoObject placeholders (OCAtari returns these for undetected object slots)
        if skip_no_object and category == "NoObject":
            continue

        # Skip if no category
        if category is None:
            continue

        # Assert no duplicate real game objects
        if assert_unique:
            assert category not in objects_map, (
                f"Duplicate game object detected: {category} already exists in map. "
                f"This indicates OCAtari returned multiple objects with the same category."
            )

        # Add object to map (for non-unique, first occurrence wins)
        if category not in objects_map:
            objects_map[category] = obj

    return objects_map
