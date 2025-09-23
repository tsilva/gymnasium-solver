from __future__ import annotations

import pytorch_lightning as pl
from typing import Callable, Optional

def linear(base_value: float, target_value: float, progress: float, target_progress: float) -> float:
    """
    Linearly interpolate from base_value to target_value as progress increases from 0 to target_progress.
    If progress >= target_progress, return target_value (no further changes).
    """
    if progress >= target_progress: return target_value
    interp = (progress / target_progress) if target_progress > 0 else 1.0
    return base_value + (target_value - base_value) * interp

SCHEDULERS_MAP = {
    "linear": linear,
}

class HyperparameterSchedulerCallback(pl.Callback):
    """Update scheduled hyperparameters (eg: policy_lr, clip_range, ent_coef) at epoch end.

    Notes:
    - The schedule is computed from the initial value observed on first use,
      avoiding compounding updates across epochs.
    """
   
    def __init__(
        self, 
        *,
        schedule: str,
        parameter: str, 
        target_value: float,
        target_progress: float,
        set_value_fn: Optional[Callable[[pl.LightningModule, float], None]] = None
    ):
        super().__init__()

        assert schedule in SCHEDULERS_MAP, f"invalid schedule: {schedule}"
        assert 0.0 <= target_progress <= 1.0, f"invalid target_progress: {target_progress}"

        self.schedule = schedule
        self.parameter = parameter
        self.target_value = target_value
        self.target_progress = target_progress
        self.set_value_fn = set_value_fn if set_value_fn is not None else self._set_parameter_value
        self._base_value: Optional[float] = None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401rn
        # Initialize the base value on first use (avoid compounding updates)
        if self._base_value is None:
            self._base_value = self._get_parameter_value(pl_module) # TODO: why this

        # TODO: ensure this still works after switching to effective steps
        # Calculate the current training progress
        progress = pl_module._calc_training_progress() # TODO: pass function to init instead

        # Retrieve the selected scheduler function
        scheduler_fn = SCHEDULERS_MAP[self.schedule]

        # Calculate the new value of the parameter based on current progress
        new_value = scheduler_fn(self._base_value, self.target_value, progress, self.target_progress)

        # Set the new value of the parameter
        self.set_value_fn(pl_module, new_value)

    def _get_parameter_value(self, pl_module: pl.LightningModule) -> float:
        assert hasattr(pl_module, self.parameter), f"module {pl_module} has no attribute {self.parameter}"
        return getattr(pl_module.config, self.parameter)

    def _set_parameter_value(self, pl_module: pl.LightningModule, value: float) -> None:
        setattr(pl_module, self.parameter, value)
