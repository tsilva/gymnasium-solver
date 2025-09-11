from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, List

from .metrics_buffer import MetricsBuffer
from .metrics_history import MetricsHistory


class MetricsRecorder:
    """
    General-purpose, in-memory metrics recorder with dynamic namespaces.

    - Buffers are created lazily per namespace (e.g., "train", "val", "eval", "test", "foo").
    - A single `MetricsHistory` is shared across all namespaces and keyed by `step_key`.
    - `update_history` expects fully-prefixed snapshots (e.g., "train/*", "val/*", ...).

    Typical usage:
        rec = MetricsRecorder(step_key="train/total_timesteps")
        rec.record("train", {"loss": 0.2, "lr": 3e-4})
        rec.record("val", {"acc": 0.83})
        epoch_train_means = rec.compute_epoch_means("train")
        rec.reset_epoch("train")
        rec.update_history({"train/total_timesteps": 1024, "train/loss": 0.2})
    """

    def __init__(self, *, step_key: str = "train/total_timesteps") -> None:
        self._buffers: MutableMapping[str, MetricsBuffer] = {}
        self._history = MetricsHistory(step_key=step_key)

    # ---- hot-path recorder ----
    def record(self, namespace: str, metrics: Mapping[str, Any]) -> None:
        """
        Log a metrics mapping into the buffer for `namespace`. The buffer is
        allocated automatically if it doesn't yet exist.
        """
        if not namespace or not metrics:
            return
        buf = self._buffers.get(namespace)
        if buf is None:
            buf = self._buffers[namespace] = MetricsBuffer()
        buf.log(metrics)

    # ---- epoch lifecycle ----
    def reset_epoch(self, namespace: str | None = None) -> None:
        """
        Clear the buffer for `namespace`. If `namespace` is None, clear all buffers.
        If the namespace doesn't exist yet, this is a no-op.
        """
        if namespace is None:
            for buf in self._buffers.values():
                buf.clear()
            return

        buf = self._buffers.get(namespace)
        if buf is not None:
            buf.clear()

    def compute_epoch_means(self, namespace: str) -> Dict[str, float]:
        """
        Compute mean values for the given `namespace`. If the namespace doesn't
        exist or has no data, an empty dict is returned.
        """
        buf = self._buffers.get(namespace)
        if buf is None:
            return {}
        return dict(buf.means())

    # ---- history ----
    def update_history(self, snapshot: Mapping[str, Any]) -> None:
        """
        Update the shared step-aware history from a fully-prefixed snapshot.
        Example:
            {"train/total_timesteps": 2048, "train/loss": 0.15, "val/acc": 0.84}
        """
        if not snapshot:
            return
        self._history.update(snapshot)

    def history(self) -> Dict[str, Any]:
        """Return the history as a plain dict."""
        return self._history.as_dict()

    # ---- introspection helpers ----
    def namespaces(self) -> List[str]:
        """Return the currently allocated namespaces (sorted)."""
        return sorted(self._buffers.keys())

    def ensure_namespace(self, namespace: str) -> None:
        """Ensure a buffer exists for `namespace` (allocated if missing)."""
        if namespace not in self._buffers:
            self._buffers[namespace] = MetricsBuffer()