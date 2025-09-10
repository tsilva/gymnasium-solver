from __future__ import annotations

from typing import Any, Dict, Mapping

from .metrics_buffer import MetricsBuffer
from .metrics_history import MetricsHistory


class MetricsRecorder:
    """
    Interface for recording training/eval metrics with stage-scoped aggregation
    and a shared step-aware history.
    """

    def record_train(self, metrics: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def record_eval(self, metrics: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def reset_epoch(self, stage: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def compute_epoch_means(self, stage: str) -> Dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def update_history(self, snapshot: Mapping[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def history(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryMetricsRecorder(MetricsRecorder):
    """
    In-memory implementation backed by per-stage buffers and a shared numeric history.

    - Two independent stage buffers: 'train' and 'eval'.
    - A single `MetricsHistory` keyed by a canonical step key (default: train/total_timesteps).
    - History is updated by passing fully-prefixed snapshots (e.g., train/*, eval/*) at flush time.
    """

    def __init__(self, *, step_key: str = "train/total_timesteps") -> None:
        self._train_buf = MetricsBuffer()
        self._eval_buf = MetricsBuffer()
        self._history = MetricsHistory(step_key=step_key)

    # ---- hot-path recorders ----
    def record_train(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return
        self._train_buf.log(metrics)

    def record_eval(self, metrics: Mapping[str, Any]) -> None:
        if not metrics:
            return
        self._eval_buf.log(metrics)

    # ---- epoch lifecycle ----
    def reset_epoch(self, stage: str) -> None:
        s = str(stage).lower()
        if s == "train":
            self._train_buf.clear()
        elif s == "val":
            self._eval_buf.clear()
        else:
            raise ValueError(f"Unknown stage for reset_epoch: {stage}")

    def compute_epoch_means(self, stage: str) -> Dict[str, float]:
        s = str(stage).lower()
        if s == "train":
            return dict(self._train_buf.means())
        if s == "val":
            return dict(self._eval_buf.means())
        raise ValueError(f"Unknown stage for compute_epoch_means: {stage}")

    # ---- history ----
    def update_history(self, snapshot: Mapping[str, Any]) -> None:
        # Expect fully-prefixed namespaced keys (e.g., train/* or eval/*)
        self._history.update(snapshot)

    def history(self) -> Dict[str, Any]:
        return self._history.as_dict()

