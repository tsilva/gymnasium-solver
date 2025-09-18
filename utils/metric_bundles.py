from typing import Callable, Iterable


class MetricMonitorBundle:
    """Interface for metric monitor bundles.

    Implementations should return an iterable of monitor callables via
    `get_monitor_fns()`. Each callable receives the metrics history dict
    and returns either a `MetricAlert`, an iterable of `MetricAlert`, or
    a false-y value when no alert should be emitted.

    The default implementation auto-registers any bound method whose
    name starts with `_monitor`.
    """

    def get_monitor_fns(self) -> Iterable[Callable]:
        fns: list[Callable] = []
        for name in dir(self):
            if not name.startswith("_monitor"):
                continue
            fn = getattr(self, name)
            if not callable(fn):
                continue
            fns.append(fn)
        return tuple(fns)

