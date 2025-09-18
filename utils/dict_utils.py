"""Small dictionary helpers used across the project."""

from typing import Any, Dict, Iterable, Mapping, List

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


def order_namespaces(
    grouped: Dict[str, Dict[str, Any]],
    preferred_order: Iterable[str] | None = None,
) -> List[str]:
    """Order namespace keys with an optional preferred prefix order.

    - If ``preferred_order`` is provided, include those namespaces first (in the
      given order) when they exist in ``grouped``.
    - Then append remaining namespaces in lexicographic order.
    - If ``preferred_order`` is falsy, simply return all namespaces sorted.
    """
    group_keys = list(grouped.keys())
    if not preferred_order:
        return sorted(group_keys)

    pref = [ns for ns in preferred_order if ns in grouped]
    rest = sorted(ns for ns in group_keys if ns not in pref)
    return pref + rest


def sort_subkeys_by_priority(
    grouped: Dict[str, Dict[str, Any]],
    ns_order: Iterable[str],
    key_priority_map: Mapping[str, int] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Return a copy of ``grouped`` with each namespace's subkeys sorted.

    Sorting respects ``key_priority_map`` where lower index wins. Keys not in
    the priority map are ordered by case-insensitive subkey.

    The priority map is indexed by full metric key (e.g., ``"ns/sub"``). For
    empty subkeys, the full key is just the namespace (matching callers).
    """

    def sort_key(namespace: str, subkey: str):
        if key_priority_map:
            full_key = f"{namespace}/{subkey}" if subkey else namespace
            idx = key_priority_map.get(full_key)
            if idx is not None:
                return (0, idx)
        return (1, subkey.lower())

    out: Dict[str, Dict[str, Any]] = dict(grouped)
    for ns in ns_order:
        sub = grouped.get(ns)
        if not sub:
            continue
        out[ns] = dict(sorted(sub.items(), key=lambda kv: sort_key(ns, kv[0])))
    return out
