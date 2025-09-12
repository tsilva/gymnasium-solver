from typing import Dict, List, Callable

class MetricsTriggers:

    def __init__(self) -> None:
        self.triggers: Dict[str, Callable] = {}

    def register_trigger(self, metric: str, trigger: Callable) -> None:
        self.triggers[metric] = trigger

    def check_triggers(self) -> None:
        alerts = {}
        for metric, trigger in self.triggers.items():
            alert = trigger()
            if alert: alerts[metric] = alert
        return alerts
    