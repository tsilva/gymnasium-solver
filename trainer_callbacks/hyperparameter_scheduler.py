from __future__ import annotations

import math
import pytorch_lightning as pl
from typing import Callable, Optional


def linear(start_value: float, end_value: float, fraction: float) -> float:
    """Linear interpolation between *start_value* and *end_value* given fraction in [0, 1]."""
    clamped = max(0.0, min(fraction, 1.0))
    return start_value + (end_value - start_value) * clamped


def cosine(start_value: float, end_value: float, fraction: float) -> float:
    """Cosine annealing from *start_value* to *end_value* given fraction in [0, 1]."""
    clamped = max(0.0, min(fraction, 1.0))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * clamped))
    return end_value + (start_value - end_value) * cosine_decay


def exponential(start_value: float, end_value: float, fraction: float) -> float:
    """Exponential decay from *start_value* to *end_value*."""
    clamped = max(0.0, min(fraction, 1.0))
    # Use decay_rate=2.0 for reasonable exponential curve
    decay_rate = 2.0
    exp_decay = math.exp(-decay_rate * clamped)
    # Normalize so that at fraction=0 we get start_value and at fraction=1 we get end_value
    exp_range = 1.0 - math.exp(-decay_rate)
    normalized_decay = (exp_decay - math.exp(-decay_rate)) / exp_range
    return end_value + (start_value - end_value) * normalized_decay


SCHEDULERS_MAP = {
    "linear": linear,
    "cosine": cosine,
    "exponential": exponential,
}

class HyperparameterSchedulerCallback(pl.Callback):
    """Update scheduled hyperparameters (eg: policy_lr, clip_range, ent_coef) at epoch end.

    Supports optional warmup: linearly increase from end_value to start_value during warmup_fraction,
    then apply the chosen scheduler from start_value to end_value for the remaining fraction.
    """

    def __init__(
        self,
        *,
        schedule: str,
        parameter: str,
        start_value: float,
        end_value: float,
        start_step: float,
        end_step: float,
        warmup_fraction: float = 0.0,
        set_value_fn: Optional[Callable[[pl.LightningModule, float], None]] = None,
    ):
        super().__init__()

        if schedule not in SCHEDULERS_MAP:
            raise ValueError(f"invalid schedule: {schedule}")
        if end_step < start_step:
            raise ValueError("schedule end_step must be >= start_step")
        if not (0.0 <= warmup_fraction < 1.0):
            raise ValueError(f"warmup_fraction must be in [0, 1), got {warmup_fraction}")

        self.parameter = parameter
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        self.warmup_fraction = warmup_fraction
        self.schedule_fn = SCHEDULERS_MAP[schedule]
        self.set_value_fn = set_value_fn if set_value_fn is not None else self._set_parameter_value

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        total_steps = self._get_total_vec_steps(pl_module)
        if total_steps is None:
            return

        fraction = self._compute_fraction(total_steps)

        # Apply warmup if configured
        if self.warmup_fraction > 0.0 and fraction < self.warmup_fraction:
            # Linear warmup from end_value to start_value
            warmup_progress = fraction / self.warmup_fraction
            new_value = self.end_value + (self.start_value - self.end_value) * warmup_progress
        else:
            # Apply scheduler (adjust fraction to account for warmup)
            if self.warmup_fraction > 0.0:
                scheduler_fraction = (fraction - self.warmup_fraction) / (1.0 - self.warmup_fraction)
            else:
                scheduler_fraction = fraction
            new_value = self.schedule_fn(self.start_value, self.end_value, scheduler_fraction)

        self.set_value_fn(pl_module, new_value)

    def _compute_fraction(self, total_steps: float) -> float:
        if total_steps <= self.start_step:
            return 0.0
        if total_steps >= self.end_step or self.end_step == self.start_step:
            return 1.0
        return (total_steps - self.start_step) / (self.end_step - self.start_step)

    def _get_total_vec_steps(self, pl_module: pl.LightningModule) -> Optional[float]:
        try:
            collector = pl_module.get_rollout_collector("train")
        except Exception:
            return None
        total_steps = getattr(collector, "total_vec_steps", None)
        if total_steps is None:
            return None
        return float(total_steps)

    def _set_parameter_value(self, pl_module: pl.LightningModule, value: float) -> None:
        pl_module.set_hyperparameter(self.parameter, value)
