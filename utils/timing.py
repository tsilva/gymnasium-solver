from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

def _now_ns() -> int:
    return time.perf_counter_ns()

@dataclass
class Marker:
    started_ns: int
    started_value: float

@dataclass
class TimingTracker:
    """Utility to track elapsed time and compute throughput of arbitrary counters.

    Backwards compatible with the previous step-only API via `steps`/`steps_now`
    and `fps_since`, which is now an alias for generic throughput.
    """

    markers: Dict[str, Marker] = field(default_factory=dict)

    def restart(
        self,
        name: str,
        *,
        value: float = 0.0,
        steps: Optional[int] = None,
        now_ns: Optional[int] = None,
    ) -> None:
        """Start (or restart) a marker and associate a baseline counter value.

        Args:
            name: Marker identifier.
            value: Baseline counter value at the start (generic, any unit).
            steps: Deprecated alias for `value` for backwards compatibility.
            now_ns: Optional timestamp to use (monotonic, in ns). Defaults to perf_counter_ns().
        """
        baseline = float(steps) if steps is not None else float(value)
        self.markers[name] = Marker(started_ns=now_ns or _now_ns(), started_value=baseline)

    def seconds_since(self, name: str, *, now_ns: Optional[int] = None) -> float:
        m = self.markers[name]
        return max(((now_ns or _now_ns()) - m.started_ns) / 1e9, 1e-12)

    def throughput_since(
        self,
        name: str,
        *,
        value_now: Optional[float] = None,
        steps_now: Optional[int] = None,
        now_ns: Optional[int] = None,
    ) -> float:
        """Compute throughput of an arbitrary counter since the marker started.

        Args:
            name: Marker identifier.
            value_now: Current value of the counter (generic, any unit).
            steps_now: Deprecated alias for `value_now` for backwards compatibility.
            now_ns: Optional timestamp to use (monotonic, in ns).

        Returns:
            Counter units per second since the marker started.
        """
        if value_now is None and steps_now is None:
            raise ValueError("throughput_since requires `value_now` (or legacy `steps_now`).")
        current_value = float(steps_now) if steps_now is not None else float(value_now)
        m = self.markers[name]
        dt = self.seconds_since(name, now_ns=now_ns)
        dvalue = max(current_value - m.started_value, 0.0)
        return float(dvalue) / dt if dt > 0 else 0.0

    def fps_since(self, name: str, *, steps_now: int, now_ns: Optional[int] = None) -> float:
        """Alias for `throughput_since` kept for backwards compatibility.

        Interprets the counter as timesteps and returns frames-per-second.
        """
        return self.throughput_since(name, value_now=float(steps_now), now_ns=now_ns)