from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from .metrics_recorder import MetricsRecorder


@dataclass(frozen=True)
class MetricAlert:
    """Represents a triggered metric alert.

    Fields
    - _id: unique identifier for the alert
    - metric: fully-qualified key (e.g., 'train/ppo/approx_kl')
    - message: short human-readable description
    - tip: optional hint to address the alert
    """
    _id: str 
    metric: str
    message: str
    tip: Optional[str] = None


class MetricsMonitor:
    """Lightweight registry of metric monitor functions with helpers for common checks.

    Monitor functions return a MetricAlert instance (or an iterable of MetricAlert)
    when a condition is met, or a false-y value otherwise. This class only stores
    and executes them; registration is up to callers.
    """

    def __init__(self, metrics_recorder: MetricsRecorder) -> None:
        # Allow multiple monitor functions per metric key
        self.metrics_recorder = metrics_recorder

        # Global monitors: functions that decide which metric they apply to.
        # Each global monitor receives (full_history) and returns MetricAlert or an iterable of MetricAlert.
        self.monitor_fns: List[Callable[[Dict[str, List[Tuple[int, float]]]], Optional[Union["MetricAlert", Iterable["MetricAlert"]]]]] = []
        self.active_alerts: Dict[str, List[MetricAlert]] = {}
        self.alerts_counter: Dict[str, Dict[str, Any]] = {}
        self.total_epochs_seen: int = 0

    def register_bundle(self, bundle: "MetricMonitorBundle") -> None:
        """Register all monitor functions from a bundle in one call.

        A bundle exposes `get_monitor_fns()` which returns an iterable of
        either (metric_key, monitor_fn) pairs (legacy) or bare monitor
        callables (preferred). Bare monitors must return MetricAlert instances
        (or iterables thereof) that contain the fully-qualified metric key.
        """
        for monitor_fn in bundle.get_monitor_fns():
            assert callable(monitor_fn), "Monitor item must be a callable or (key, fn) tuple"
            self.monitor_fns.append(monitor_fn)

    # ----- execution -----
    def check(self, epoch: Optional[int] = None) -> Dict[str, List[str]]:
        """Execute monitor functions and return a mapping of key -> list of alert messages."""

        # Collect alerts for each metric
        metric_alerts_map: Dict[str, List[MetricAlert]] = {}

        if epoch is not None:
            self.total_epochs_seen = max(self.total_epochs_seen, epoch + 1)

        # 1) Global monitors
        history = self.metrics_recorder.history()
        for fn in self.monitor_fns:
            alerts = fn(history)
            if not alerts: continue
            if not isinstance(alerts, list): alerts = [alerts]
            for alert in alerts:
                metric_alerts_map.setdefault(alert.metric, []).append(alert)
                counter = self.alerts_counter.setdefault(
                    alert._id,
                    dict(alert=alert, count=0, epochs=set())
                )
                counter["count"] += 1
                epochs: Set[int] = counter.setdefault("epochs", set())
                if epoch is not None:
                    epochs.add(epoch)
            
        # Compute the new set of active alerts strictly from this check().
        # Any metric not re-emitting an alert in the current epoch is cleared.
        new_active: Dict[str, List[MetricAlert]] = {
            metric: alerts for metric, alerts in metric_alerts_map.items() if alerts
        }

        # Derive added/removed by diffing against the previous snapshot
        prev_keys = set(self.active_alerts.keys())
        new_keys = set(new_active.keys())
        added_keys = list(new_keys - prev_keys)
        removed_keys = list(prev_keys - new_keys)

        # Swap in the fresh snapshot
        self.active_alerts = new_active

        # Return active/added/removed alerts (by metric key)
        return dict(
            active=list(new_keys),
            added=added_keys,
            removed=removed_keys,
        )

    def get_active_alerts(self) -> Dict[str, List["MetricAlert"]]:
        return dict(self.active_alerts)

    def get_alerts_by_frequency(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for data in self.alerts_counter.values():
            epochs_set: Set[int] = data.get("epochs") or set()
            summaries.append(
                dict(
                    alert=data["alert"],
                    count=data["count"],
                    epoch_count=len(epochs_set),
                    epochs=tuple(sorted(epochs_set)),
                )
            )

        return sorted(
            summaries,
            key=lambda x: (x["epoch_count"], x["count"]),
            reverse=True,
        )

    def get_total_epochs(self) -> int:
        return self.total_epochs_seen
