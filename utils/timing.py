# utils/timing.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

def _now_ns() -> int:
    return time.perf_counter_ns()

@dataclass
class Marker:
    started_ns: int
    started_steps: int

@dataclass
class TimingTracker:
    """Generic timing + throughput helper.
    Create named markers, then query elapsed seconds or FPS since that marker.
    """
    markers: Dict[str, Marker] = field(default_factory=dict)

    def restart(self, name: str, *, steps: int = 0, now_ns: Optional[int] = None) -> None:
        self.markers[name] = Marker(started_ns=now_ns or _now_ns(), started_steps=int(steps))

    def seconds_since(self, name: str, *, now_ns: Optional[int] = None) -> float:
        m = self.markers[name]
        return max(((now_ns or _now_ns()) - m.started_ns) / 1e9, 1e-12)

    def fps_since(self, name: str, *, steps_now: int, now_ns: Optional[int] = None) -> float:
        m = self.markers[name]
        dt = self.seconds_since(name, now_ns=now_ns)
        dsteps = max(int(steps_now) - m.started_steps, 0)
        return float(dsteps) / dt if dt > 0 else 0.0

    # Convenience: create once if absent
    def ensure(self, name: str, *, steps: int = 0) -> None:
        if name not in self.markers:
            self.restart(name, steps=steps)