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
    """Utility to track elapsed time and compute throughput of arbitrary counters.

    Backwards compatible with the previous step-only API via `steps`/`steps_now`
    and `throughput_since`, which is now an alias for generic throughput.
    """

    markers: Dict[str, Marker] = field(default_factory=dict)

    def start(
        self,
        marker_id: str,
        *,
        values: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Start a marker and associate a baseline counter value.

        Args:   
            name: Marker identifier.
            values: Baseline counter value at the start (generic, any unit).    
            values: Mapping of baseline counters for multi-metric throughput.
        """
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
        """Compute throughput of an arbitrary counter since the marker started.

        Args:
            marker_id: Marker identifier.
            values_now: Current value of the counter (generic, any unit).

        Returns:
            Counter units per second since the marker started (float or dict).
        """
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
