"""Early stopping callback, decoupled from evaluation.

This callback inspects the latest metrics exposed by the Trainer and decides
when to stop training based on a simple threshold rule. By default, it stops
when the cumulative timesteps reach a configured limit.
"""

from __future__ import annotations

import pytorch_lightning as pl

class EarlyStoppingCallback(pl.Callback):
    """Generic early-stopping via metric threshold.

    Args:
        metric_key: Full metric name to monitor (e.g., "train/total_timesteps").
        mode: One of {"max", "min"}. For "max", training stops when value >= threshold.
              For "min", training stops when value <= threshold.
        threshold: Numeric threshold to trigger stop. If None, callback is inert.
        verbose: If True, prints a stop reason when triggered.
    """

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
        # If value is not available yet, do nothing
        value = trainer.logged_metrics.get(self.metric_key)
        if value is None: return

        # If threshold wasn't reached yet then return (do nothing)
        _should_stop_max = self.mode == "max" and value >= self.threshold
        _should_stop_min = self.mode == "min" and value <= self.threshold
        should_stop = _should_stop_max or _should_stop_min
        if not should_stop: return
        
        # Threshold reached, signal Trainer to stop
        trainer.should_stop = True
        
        # Print reason if verbose
        comp_op = ">=" if self.mode == "max" else "<="
        print(f"Early stopping! '{self.metric_key}': {value} {comp_op} {self.threshold}.")
