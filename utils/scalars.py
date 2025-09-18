"""Scalar conversion helpers used across the project.

Provides utilities to coerce framework-specific numeric types (NumPy, PyTorch)
into basic Python scalars when possible, and to filter dictionaries down to
scalar-valued entries for safe serialization/logging.
"""

from __future__ import annotations

import numbers
from typing import Any, Dict

import numpy as np
import torch


def to_scalar(x: Any) -> Any:
    """Return a Python scalar (int/float/bool) if x is scalar-like, else None.

    Rules:
    - Built-in numeric types (including bool) are returned unchanged.
    - NumPy scalars -> Python scalars via ``.item()``.
    - NumPy arrays with exactly one element -> that element via ``.item()``.
    - PyTorch tensors with exactly one element -> ``.item()``.
    - As a last resort, attempt ``float(x)`` when it clearly behaves like a number.
      If that fails, return ``None``.
    """
    # Built-in numeric types are fine as-is
    if isinstance(x, numbers.Number) or isinstance(x, bool):
        return x

    # NumPy scalar (e.g., np.float32(3.14)) -> Python scalar
    if isinstance(x, np.generic):
        return x.item()

    # NumPy array with exactly one element
    if isinstance(x, np.ndarray):
        if x.ndim == 0 or x.size == 1:
            return x.reshape(()).item()
        return None  # non-scalar -> skip

    # PyTorch tensor with exactly one element
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return None  # non-scalar -> skip

    # Anything else: try a last-resort cast to float if it clearly acts numeric
    try:
        return float(x)
    except Exception:
        return None


def only_scalar_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the dict keeping only entries whose values are scalars.

    Values are converted to basic Python scalar types when possible using
    ``to_scalar``; non-convertible values are dropped.
    """
    cleaned: Dict[str, Any] = {}
    for k, v in d.items():
        sv = to_scalar(v)
        if sv is not None:
            cleaned[k] = sv
    return cleaned

