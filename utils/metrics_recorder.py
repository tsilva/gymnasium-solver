from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping

from .metrics_buffer import MetricsBuffer
from .metrics_history import MetricsHistory
from .scalars import only_scalar_values


class MetricsRecorder:
    """In-memory metrics recorder with dynamic namespaces and shared step history."""

    def __init__(self, step_key: str | None = None) -> None:
        self._buffers: MutableMapping[str, MetricsBuffer] = {}
        self._history = MetricsHistory()
        # Optional logical step key retained for backward compatibility with
        # callers that provide it; current implementation infers steps from
        # the snapshot keys themselves.
        self.step_key = step_key

    # ---- hot-path recorder ----
    def record(self, namespace: str, metrics: Mapping[str, Any]) -> None:
        """Record metrics to a namespace; allocates the buffer if missing."""
        assert len(metrics) > 0, "metrics cannot be empty"

        # Convert non-scalar values to scalars (eg: numpy, torch, etc.)
        metrics = only_scalar_values(metrics)

        # Assert values are valid before logging (eg: numeric, not NaN, not Inf)
        self._assert_valid_values(metrics)

        buffer = self._ensure_buffer(namespace)
        buffer.log(metrics)

    # ---- epoch lifecycle ----
    def reset_epoch(self, namespace: str) -> None:
        """Clear the namespace buffer (no-op if missing)."""
        buffer = self._ensure_buffer(namespace)
        buffer.clear()

    def compute_epoch_means(self, namespace: str) -> Dict[str, float]:
        """Compute mean values for a namespace; returns {} when empty."""
        buffer = self._ensure_buffer(namespace)
        means = dict(buffer.means())
        return means

    # ---- history ----
    def update_history(self, snapshot: Mapping[str, Any]) -> None:
        """Update step-aware history from a fully-prefixed snapshot."""
        assert len(snapshot) > 0, "snapshot cannot be empty"

        self._history.update(snapshot)

    def history(self) -> Dict[str, Any]:
        """Return the history as a plain dict."""
        return self._history.as_dict()

    # ---- introspection helpers ----
    def namespaces(self) -> List[str]:
        """Return the currently allocated namespaces (sorted)."""
        return sorted(self._buffers.keys())

    def _ensure_buffer(self, namespace: str) -> None:
        """Ensure a buffer exists for `namespace` (allocated if missing)."""
        assert len(namespace) > 0, "namespace cannot be empty"
        if namespace in self._buffers: return self._buffers[namespace]
        buffer = MetricsBuffer()
        self._buffers[namespace] = buffer
        return buffer

    def _assert_valid_values(self, metrics: Mapping[str, Any]) -> None:
        for metric, value in metrics.items():
            assert isinstance(value, (int, float)), f"metric '{metric}' is not a number: {value}"
            assert not math.isnan(value), f"metric '{metric}' is NaN: {value}"
            assert not math.isinf(value), f"metric '{metric}' is Inf: {value}"
