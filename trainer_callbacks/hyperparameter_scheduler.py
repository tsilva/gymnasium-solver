from __future__ import annotations

import pytorch_lightning as pl
from typing import Callable, Optional

def linear(base_value: float, progress: float) -> float:
    return max(base_value * (1.0 - progress), 0.0)

SCHEDULERS_MAP = {
    "linear": linear,
}

class HyperparameterSchedulerCallback(pl.Callback):
    """Update scheduled hyperparameters (eg: policy_lr, clip_range, ent_coef) at epoch end."""
   
    def __init__(
        self, 
        schedule: str,
        parameter: str, 
        getter_fn: Optional[Callable[[pl.LightningModule], float]] = None, 
        setter_fn: Optional[Callable[[pl.LightningModule, float], None]] = None
    ):
        super().__init__()

        self.schedule = schedule
        self.parameter = parameter
        self.getter_fn = getter_fn if getter_fn is not None else self._get_parameter_value
        self.setter_fn = setter_fn if setter_fn is not None else self._set_parameter_value

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401rn
        # Retrieve the current value of the parameter
        parameter_value = self.getter_fn(pl_module)

        # Calculate the current training progress
        progress = pl_module._calc_training_progress()

        # Retrieve the selected scheduler function
        scheduler_fn = SCHEDULERS_MAP[self.schedule]

        # Calculate the new value of the parameter based of current progress
        new_value = scheduler_fn(parameter_value, progress)

        # Set the new value of the parameter
        self.setter_fn(pl_module, new_value)

    def _get_parameter_value(self, pl_module: pl.LightningModule) -> float:
        assert hasattr(pl_module, self.parameter), f"Module {pl_module} has no attribute {self.parameter}"
        return getattr(pl_module, self.parameter)

    def _set_parameter_value(self, pl_module: pl.LightningModule, value: float) -> None:
        setattr(pl_module, self.parameter, value)