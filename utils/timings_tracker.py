from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union


def _now_ns() -> int:
    return time.perf_counter_ns()

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float))

def numeric_values(values: Mapping[str, Any]) -> Dict[str, float]:
    numeric_values = {}
    for k, v in values.items():
        if not is_number(v): continue
        numeric_values[k] = float(v)
    return numeric_values

@dataclass
class Marker:
    started_ns: int
    started_values: Dict[str, float]

@dataclass
class TimingsTracker:
    """Track elapsed time and compute per-second throughput for arbitrary counters."""

    markers: Dict[str, Marker] = field(default_factory=dict)

    def start(
        self,
        marker_id: str,
        *,
        values: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Start a marker and record baseline counter values."""
        marker_ns = _now_ns()

        # Discard non-numeric values
        values_numeric = numeric_values(values)

        self.markers[marker_id] = Marker(
            started_ns=marker_ns,
            started_values=values_numeric,
        )

    def seconds_since(self, marker_id: str) -> float:
        m = self.markers[marker_id]
        return max((_now_ns() - m.started_ns) / 1e9, 1e-12)

    def throughput_since(
        self,
        marker_id: str,
        *,
        values_now: Mapping[str, Any],
    ) -> Union[float, Dict[str, float]]:
        """Compute per-second throughput since marker start from current counters."""
        marker = self.markers[marker_id]

        values_now_numeric = numeric_values(values_now)
        
        elapsed = self.seconds_since(marker_id)
        started_values = marker.started_values
        throughput: Dict[str, float] = {}
        for key, current in values_now_numeric.items():
            start_value = started_values.get(key, 0.0)
            delta = current - start_value
            throughput[key] = max(delta, 0.0) / elapsed
        return throughput
