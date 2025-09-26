from __future__ import annotations

import pytorch_lightning as pl
from typing import Callable, Optional


def linear(start_value: float, end_value: float, fraction: float) -> float:
    """Linear interpolation between *start_value* and *end_value* given fraction in [0, 1]."""
    clamped = max(0.0, min(fraction, 1.0))
    return start_value + (end_value - start_value) * clamped

SCHEDULERS_MAP = {
    "linear": linear,
}

class HyperparameterSchedulerCallback(pl.Callback):
    """Update scheduled hyperparameters (eg: policy_lr, clip_range, ent_coef) at epoch end."""

    def __init__(
        self,
        *,
        schedule: str,
        parameter: str,
        start_value: float,
        end_value: float,
        start_step: float,
        end_step: float,
        set_value_fn: Optional[Callable[[pl.LightningModule, float], None]] = None,
    ):
        super().__init__()

        if schedule not in SCHEDULERS_MAP:
            raise ValueError(f"invalid schedule: {schedule}")
        if end_step < start_step:
            raise ValueError("schedule end_step must be >= start_step")

        self.parameter = parameter
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        self.schedule_fn = SCHEDULERS_MAP[schedule]
        self.set_value_fn = set_value_fn if set_value_fn is not None else self._set_parameter_value

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        total_steps = self._get_total_vec_steps(pl_module)
        if total_steps is None:
            return

        fraction = self._compute_fraction(total_steps)
        new_value = self.schedule_fn(self.start_value, self.end_value, fraction)
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
        setattr(pl_module, self.parameter, value)
        if hasattr(pl_module.config, self.parameter):
            setattr(pl_module.config, self.parameter, value)
