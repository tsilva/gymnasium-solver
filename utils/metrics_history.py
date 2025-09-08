from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple


class MetricsHistory:
    """
    Lightweight numeric metrics history for terminal summaries.

    Tracks numeric values over time keyed by metric name, storing
    (step, value) pairs. The canonical step is inferred from a
    designated step key (default: "train/total_timesteps").
    """

    def __init__(self, step_key: str = "train/total_timesteps") -> None:
        self._history: Dict[str, List[Tuple[int, float]]] = {}
        self._last_step: int = 0
        self._step_key: str = step_key

    def update(self, metrics: Mapping[str, Any]) -> None:
        """
        Ingest a mapping of metrics and append numeric values to history.

        - Updates last known step from the configured step key when present.
        - Skips non-numeric values and keys ending with "action_dist".
        - Uses the current last step for all metrics except the step key
          itself, which uses its own value as the step.
        """
        # Update last step if step key present and numeric
        step_val = metrics.get(self._step_key)
        if isinstance(step_val, (int, float)):
            self._last_step = int(step_val)

        for k, v in metrics.items():
            if k.endswith("action_dist"):
                continue
            if not isinstance(v, (int, float)):
                continue
            step = int(v) if k == self._step_key else self._last_step
            self._history.setdefault(k, []).append((step, float(v)))

    def as_dict(self) -> Dict[str, List[Tuple[int, float]]]:
        """Return the internal history mapping (metric -> list[(step, value)])."""
        return self._history

    def clear(self) -> None:
        """Clear all recorded history."""
        self._history.clear()

    def __bool__(self) -> bool:  # truthiness: empty vs non-empty
        return any(self._history.values())

