from typing import Dict, Callable, Optional, List

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
        self.monitor_fns: Dict[str, List[Callable[[], Optional[str]]]] = {}
        self.active_alerts: Dict[str, List[str]] = {}

    # ----- registration -----
    def register(self, key: str, monitor_fn: Callable[[], Optional[str]]) -> None:
        """Register a monitor function for a fully-qualified metric key (e.g., train/approx_kl).

        Multiple monitor functions can be registered for the same metric key.
        """
        self.monitor_fns.setdefault(key, []).append(monitor_fn)

    # ----- execution -----
    def check(self) -> Dict[str, List[str]]:
        """Execute monitor functions and return a mapping of key -> list of alert messages."""

        # Collect alerts for each metric
        metrics_alerts: Dict[str, List[str]] = {}
        for metric, fns in self.monitor_fns.items():
            # If no history for metric, skip (nothing to check)
            history = self.metrics_recorder.history()
            metric_values = history.get(metric)
            if not metric_values: continue
            
            # Execute monitor functions for each metric
            for fn in fns:
                # If no alert triggered, skip
                msg = fn(metric, metric_values)
                if not msg:  continue

                # Add alert to list
                metrics_alerts.setdefault(metric, []).append(msg)

        # For each metric, add to active alerts if alerts
        # are present, if no alerts, remove previous alerts
        for metric, alerts in metrics_alerts.items():
            if alerts: self.active_alerts[metric] = alerts
            else: del self.active_alerts[metric]

        # Return active alerts (copy to avoid mutation)
        active_alerts = self.get_active_alerts()
        return active_alerts

    def get_active_alerts(self) -> Dict[str, List[str]]:
        return dict(self.active_alerts)
