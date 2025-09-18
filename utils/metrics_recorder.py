from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, List

from .metrics_buffer import MetricsBuffer
from .metrics_history import MetricsHistory
from .scalars import only_scalar_values


class MetricsRecorder:
    """
    General-purpose, in-memory metrics recorder with dynamic namespaces.

    - Buffers are created lazily per namespace (e.g., "train", "val", "eval", "test", "foo").
    - A single `MetricsHistory` is shared across all namespaces and keyed by `step_key`.
    - `update_history` expects fully-prefixed snapshots (e.g., "train/*", "val/*", ...).

    Typical usage:
        rec = MetricsRecorder()  # uses global step key from metrics config
        rec.record("train", {"loss": 0.2, "lr": 3e-4})
        rec.record("val", {"acc": 0.83})
        epoch_train_means = rec.compute_epoch_means("train")
        rec.reset_epoch("train")
        rec.update_history({"train/total_timesteps": 1024, "train/loss": 0.2})
    """

    def __init__(self, *, step_key: str | None = None) -> None:
        self._buffers: MutableMapping[str, MetricsBuffer] = {}
        self._history = MetricsHistory(step_key=step_key)

    # ---- hot-path recorder ----
    def record(self, namespace: str, metrics: Mapping[str, Any]) -> None:
        """
        Log a metrics mapping into the buffer for `namespace`. The buffer is
        allocated automatically if it doesn't yet exist.
        """
        assert len(metrics) > 0, "metrics cannot be empty"
        buffer = self._ensure_buffer(namespace)
        metrics = only_scalar_values(metrics)
        buffer.log(metrics)

    # ---- epoch lifecycle ----
    def reset_epoch(self, namespace: str) -> None:
        """
        Clear the buffer for `namespace`.
        If the namespace doesn't exist yet, this is a no-op.
        """
        buffer = self._ensure_buffer(namespace)
        buffer.clear()

    def compute_epoch_means(self, namespace: str) -> Dict[str, float]:
        """
        Compute mean values for the given `namespace`. If the namespace doesn't
        exist or has no data, an empty dict is returned.
        """
        buffer = self._ensure_buffer(namespace)
        means = dict(buffer.means())
        return means

    # ---- history ----
    def update_history(self, snapshot: Mapping[str, Any]) -> None:
        """
        Update the shared step-aware history from a fully-prefixed snapshot.
        Example:
            {"train/total_timesteps": 2048, "train/loss": 0.15, "val/acc": 0.84}
        """
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
