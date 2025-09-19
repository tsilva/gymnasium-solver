"""Simple hyperparameter scheduling utilities.

Currently supports a linear decay schedule used for learning rate and
PPO clip range. Centralizing math helps keep callbacks and agents in sync.
"""

from __future__ import annotations

from typing import Callable, Optional


def linear(base_value: float, progress: float) -> float:
    """Linearly decay from base_value → 0 as progress goes 0 → 1.

    Clamps at 0 to avoid negative values.
    """
    return max(base_value * (1.0 - float(progress)), 0.0)


def resolve(schedule: Optional[str]) -> Optional[Callable[[float, float], float]]:
    """Return a schedule function for the given identifier, or None.

    Recognized schedules:
      - "linear": linear decay to 0 over progress in [0, 1].
    """
    if schedule is None:
        return None
    key = schedule.strip().lower()
    if key == "linear":
        return linear
    return None

