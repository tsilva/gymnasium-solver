from typing import Callable, Dict, List, Optional, Iterable, Tuple, Any

from .metrics_recorder import MetricsRecorder


class MetricsMonitor:
    """Lightweight registry of metric monitor functions with helpers for common checks.

    Monitor functions are callables returning either a string message (when the
    condition is met) or a false-y value otherwise. This class only stores and
    executes them; registration is up to callers.
    """

    def __init__(self, metrics_recorder: MetricsRecorder) -> None:
        # Allow multiple monitor functions per metric key
        self.metrics_recorder = metrics_recorder
        # Map metric key -> list of monitor functions (legacy keyed monitors)
        # Each keyed monitor receives (metric_key, full_history)
        self.monitor_fns: Dict[str, List[Callable[[str, Dict[str, List[Tuple[int, float]]]], Optional[dict]]]] = {}
        # Global monitors: functions that decide which metric they apply to.
        # Each global monitor receives (full_history) and returns an alert dict
        # containing a 'metric' field with the fully-qualified metric key.
        self.global_monitor_fns: List[Callable[[Dict[str, List[Tuple[int, float]]]], Optional[dict]]] = []
        self.active_alerts: Dict[str, List[str]] = {}
        self.alerts_counter: Dict[str, int] = {}

    # ----- registration -----
    def register(self, key: str, monitor_fn: Callable[[str, Dict[str, List[Tuple[int, float]]]], Optional[dict]]) -> None:
        """Register a monitor function for a fully-qualified metric key (e.g., train/approx_kl).

        Multiple monitor functions can be registered for the same metric key.
        """
        self.monitor_fns.setdefault(key, []).append(monitor_fn)

    def register_fn(self, monitor_fn: Callable[[Dict[str, List[Tuple[int, float]]]], Optional[dict]]) -> None:
        """Register a global monitor function that determines its own metric key.

        Global monitor functions must return an alert dict that includes a
        'metric' field specifying the fully-qualified metric key.
        """
        self.global_monitor_fns.append(monitor_fn)

    def register_bundle(self, bundle: "MetricMonitorBundle") -> None:
        """Register all monitor functions from a bundle in one call.

        A bundle exposes `get_monitor_fns()` which returns an iterable of
        either (metric_key, monitor_fn) pairs (legacy) or bare monitor
        callables (preferred). Bare monitors must return alerts containing
        a 'metric' field.
        """
        for item in bundle.get_monitor_fns():
            # Support both (key, fn) and bare fn styles
            if isinstance(item, tuple) and len(item) == 2 and callable(item[1]):
                key, fn = item
                self.register(key, fn)
            else:
                fn = item  # type: ignore[assignment]
                assert callable(fn), "Monitor item must be a callable or (key, fn) tuple"
                self.register_fn(fn)  # type: ignore[arg-type]

    # ----- execution -----
    def check(self) -> Dict[str, List[str]]:
        """Execute monitor functions and return a mapping of key -> list of alert messages."""

        # Collect alerts for each metric
        alerts_by_metric: Dict[str, List[dict]] = {}
        history = self.metrics_recorder.history()

        # 1) Global monitors
        for fn in self.global_monitor_fns:
            alert_obj = fn(history)
            for alert in _iter_alerts(alert_obj):
                metric_key = alert.get("metric")
                if not metric_key:
                    continue
                alerts_by_metric.setdefault(metric_key, []).append(alert)
                self.alerts_counter[metric_key] = self.alerts_counter.get(metric_key, 0) + 1

        # 2) Legacy keyed monitors
        for metric, fns in self.monitor_fns.items():
            for fn in fns:
                try:
                    alert_obj = fn(metric, history)
                except TypeError:
                    # Fallback to signature (history)
                    alert_obj = fn(history)  # type: ignore[misc]
                for alert in _iter_alerts(alert_obj):
                    metric_key = alert.get("metric", metric)
                    alerts_by_metric.setdefault(metric_key, []).append(alert)
                    self.alerts_counter[metric_key] = self.alerts_counter.get(metric_key, 0) + 1

        # For each metric, add to active alerts if alerts
        # are present, if no alerts, remove previous alerts
        add_alerts = {}
        remove_alerts = []
        for metric, alerts in alerts_by_metric.items():
            if alerts: add_alerts[metric] = alerts
            else: remove_alerts.append(metric)

        # Remove alerts that are no longer present
        for metric in remove_alerts: del self.active_alerts[metric]

        # Add new alerts
        self.active_alerts.update(add_alerts)   

        # Return active/added/removed alerts
        return dict(
            active=list(self.active_alerts.keys()),
            added=list(add_alerts.keys()),
            removed=list(remove_alerts)
        )

    def get_active_alerts(self) -> Dict[str, List[str]]:
        return dict(self.active_alerts)

    def get_alerts_counter(self) -> Dict[str, int]:
        return dict(self.alerts_counter)


def _iter_alerts(alert_obj: Any) -> Iterable[dict]:
    """Normalize a monitor return into an iterable of alert dicts.

    Accepts None, a single dict, or an iterable of dicts.
    """
    if not alert_obj:
        return ()
    if isinstance(alert_obj, dict):
        return (alert_obj,)
    # Best-effort: assume iterable of dicts
    try:
        return tuple(x for x in alert_obj if isinstance(x, dict))
    except TypeError:
        return ()
