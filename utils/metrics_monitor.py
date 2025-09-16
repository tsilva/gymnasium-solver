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

    # ----- registration -----
    def register(self, key: str, monitor_fn: Callable[[], Optional[str]]) -> None:
        """Register a monitor function for a fully-qualified metric key (e.g., train/approx_kl).

        Multiple monitor functions can be registered for the same metric key.
        """
        self.monitor_fns.setdefault(key, []).append(monitor_fn)

    # ----- execution -----
    def check(self) -> Dict[str, List[str]]:
        """Execute monitor functions and return a mapping of key -> list of alert messages."""
        out: Dict[str, List[str]] = {}
        for metric, fns in self.monitor_fns.items():
            history = self.metrics_recorder.history()
            metric_values = history.get(metric)
            if not metric_values: continue
            for fn in fns:
                msg = fn(metric, metric_values)
                if msg: out.setdefault(metric, []).append(msg)
        return out
