from typing import Dict, Callable, Optional, Any, Tuple, List


class MetricsTriggers:
    """Lightweight registry of metric triggers with helpers for common checks.

    Triggers are callables returning either a string message (when the
    condition is met) or a false-y value otherwise. This class only stores and
    executes them; registration is up to callers.
    """

    def __init__(self) -> None:
        # Allow multiple triggers per metric key
        self.triggers: Dict[str, List[Callable[[], Optional[str]]]] = {}

    # ----- registration -----
    def register_trigger(self, key: str, trigger: Callable[[], Optional[str]]) -> None:
        """Register a trigger for a fully-qualified metric key (e.g., train/approx_kl).

        Multiple triggers can be registered for the same metric key.
        """
        self.triggers.setdefault(key, []).append(trigger)

    # ----- execution -----
    def check_triggers(self) -> Dict[str, List[str]]:
        """Execute triggers and return a mapping of key -> list of alert messages."""
        out: Dict[str, List[str]] = {}
        for metric, fns in self.triggers.items():
            for fn in list(fns):
                try:
                    msg = fn()
                except Exception:
                    msg = None
                if msg:
                    out.setdefault(metric, []).append(str(msg))
        return out
