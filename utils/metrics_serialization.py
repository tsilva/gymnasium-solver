"""Serialize metrics dicts with configured precision and stable key ordering."""

from __future__ import annotations

import numbers
from collections import OrderedDict
from typing import Any, Dict, Iterable

from utils.dict_utils import (
    group_by_namespace as _group_by_namespace,
)
from utils.dict_utils import (
    order_namespaces as _order_namespaces,
)
from utils.dict_utils import (
    sort_subkeys_by_priority as _sort_subkeys_by_priority,
)
from utils.metrics_config import metrics_config


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Number)


def _coerce_by_precision(metric_key_or_bare: str, value: Any) -> Any:
    """Coerce numeric values per metrics.yaml precision; pass through non-numerics."""
    if value is None or not _is_number(value):
        return value

    try:
        precision = int(metrics_config.precision_for_metric(metric_key_or_bare))
    except Exception:
        precision = 2

    if precision == 0:
        try:
            return int(round(float(value)))
        except Exception:
            return int(float(value))
    else:
        try:
            return round(float(value), precision)
        except Exception:
            return float(value)


def prepare_metrics_for_json(
    metrics: Dict[str, Any],
    *,
    namespace_order: Iterable[str] = ("train", "val", "test"),
) -> Dict[str, Any]:
    """Return a new dict with values coerced by precision and keys ordered.

    Ordering respects key_priority from metrics.yaml expanded across the
    common namespaces. Keys not in the priority list are ordered by
    case-insensitive subkey within their namespace. Namespaces themselves
    are ordered per ``namespace_order`` then lexicographically.
    """
    if not metrics:
        return {}

    # If there are no namespaced keys at all, order by bare priority first,
    # then by case-insensitive key for the rest.
    if not any("/" in k for k in metrics.keys()):
        priority = tuple(metrics_config.key_priority() or ())
        pr_index = {name: idx for idx, name in enumerate(priority)}
        def sort_key(k: str) -> tuple[int, int | str]:
            return (0, pr_index[k]) if k in pr_index else (1, k.lower())
        ordered = OrderedDict()
        for k in sorted(metrics.keys(), key=sort_key):
            ordered[k] = _coerce_by_precision(k, metrics[k])
        return dict(ordered)

    # Group by namespace and determine namespace order
    grouped = _group_by_namespace(metrics)
    ns_order = _order_namespaces(grouped, namespace_order)

    # Build priority map for full keys across common namespaces
    priority = tuple(metrics_config.key_priority() or ())
    namespaces = ("train", "val", "test")
    key_priority_map = {
        f"{ns}/{sub}": idx
        for idx, sub in enumerate(priority)
        for ns in namespaces
    }

    # Sort subkeys within each namespace per priority
    sorted_grouped = _sort_subkeys_by_priority(grouped, ns_order, key_priority_map)

    # Flatten back to a flat dict preserving the order we iterate
    ordered: "OrderedDict[str, Any]" = OrderedDict()
    for ns in ns_order:
        sub = sorted_grouped.get(ns) or {}
        for subkey, value in sub.items():
            full_key = f"{ns}/{subkey}" if subkey else ns
            ordered[full_key] = _coerce_by_precision(full_key, value)

    return dict(ordered)
