"""Small dictionary helpers used across the project."""

from typing import Any, Dict

def prefix_dict_keys(data: dict, prefix: str) -> dict:
    """Return a copy of data with keys prefixed by '<prefix>/'."""
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}


def group_by_namespace(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Group flat namespaced keys by their first path segment.

    Example: {"train/acc": 1, "val/loss": 2, "epoch": 3}
      -> {"train": {"acc": 1}, "val": {"loss": 2}, "epoch": {"": 3}}
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if "/" in k:
            ns, sub = k.split("/", 1)
        else:
            ns, sub = k, ""
        grouped.setdefault(ns, {})[sub] = v
    return grouped
