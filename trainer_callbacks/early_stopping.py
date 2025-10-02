"""Early stopping callback for threshold-based stop conditions (decoupled from eval).

If `threshold` is None, the callback becomes a no-op. This allows callers to
conditionally attach the callback without having to special-case missing
thresholds upstream.
"""

from __future__ import annotations

import pytorch_lightning as pl

from utils.formatting import format_metric_value


class EarlyStoppingCallback(pl.Callback):
    """Early-stop when a monitored metric crosses a threshold (min/max)."""

    def __init__(
        self,
        metric_key: str,
        threshold: float,
        mode: str = "max",
    ) -> None:
        super().__init__()
        assert mode in {"max", "min"}, "mode must be 'max' or 'min'"
        self.metric_key = metric_key
        self.mode = mode
        self.threshold = threshold

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._maybe_stop(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._maybe_stop(trainer, pl_module)

    def _maybe_stop(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Trainer already signaled to stop, nothing to do
        if trainer.should_stop: return

        # If no threshold was provided, this callback is disabled
        if self.threshold is None: return
        
        # If value is not available yet, do nothing
        value = trainer.logged_metrics.get(self.metric_key)
        if value is None: return
        
        # If threshold wasn't reached yet then return (do nothing)
        _should_stop_max = self.mode == "max" and float(value) >= self.threshold
        _should_stop_min = self.mode == "min" and float(value) <= self.threshold
        should_stop = _should_stop_max or _should_stop_min
        if not should_stop: return

        # Threshold reached, signal Trainer to stop
        trainer.should_stop = True

        # Print reason with metrics.yaml-based formatting
        comp_op = ">=" if self.mode == "max" else "<="
        value_s = format_metric_value(self.metric_key, float(value))
        threshold_s = format_metric_value(self.metric_key, float(self.threshold))
        early_stop_reason = f"'{self.metric_key}': {value_s} {comp_op} {threshold_s}."
        print(f"Early stopping! {early_stop_reason}")

        # Store the reason in the module so that it is
        # available for the end of training report
        pl_module._early_stop_reason = early_stop_reason
