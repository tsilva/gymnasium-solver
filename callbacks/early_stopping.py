"""Early stopping callback, decoupled from evaluation.

This callback inspects the latest metrics exposed by the Trainer and decides
when to stop training based on a simple threshold rule. By default, it stops
when the cumulative timesteps reach a configured limit.
"""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch


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
        metric_key: str = "train/total_timesteps",
        mode: str = "max",
        threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        assert mode in {"max", "min"}, "mode must be 'max' or 'min'"
        self.metric_key = metric_key
        self.mode = mode
        self.threshold = threshold
        self.verbose = verbose

    # ----- Lightning hooks -----
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # No-op if not configured
        if self.threshold is None:
            return

        metrics = self._collect_metrics(trainer)
        if self.metric_key not in metrics:
            return

        val = self._to_number(metrics[self.metric_key])
        if val is None:
            return

        should_stop = (
            (self.mode == "max" and float(val) >= float(self.threshold))
            or (self.mode == "min" and float(val) <= float(self.threshold))
        )
        if should_stop:
            if self.verbose:
                print(
                    f"Early stopping: '{self.metric_key}' reached {val} (threshold={self.threshold}, mode={self.mode})."
                )
            # Signal PL to stop after this epoch
            trainer.should_stop = True

    # ----- helpers -----
    def _collect_metrics(self, trainer: "pl.Trainer") -> Dict[str, Any]:
        """Merge most recent metrics from trainer into a plain dict."""
        combo: Dict[str, Any] = {}

        dicts = []
        if hasattr(trainer, "logged_metrics") and isinstance(trainer.logged_metrics, dict):
            dicts.append(trainer.logged_metrics)
        if hasattr(trainer, "callback_metrics") and isinstance(trainer.callback_metrics, dict):
            dicts.append(trainer.callback_metrics)
        if hasattr(trainer, "progress_bar_metrics") and isinstance(trainer.progress_bar_metrics, dict):
            dicts.append(trainer.progress_bar_metrics)

        for d in dicts:
            for k, v in d.items():
                combo[k] = self._to_python_scalar(v)

        # Remove common bookkeeping keys if present
        for k in ("epoch", "step", "global_step"):
            combo.pop(k, None)
        return combo

    def _to_python_scalar(self, x: Any) -> Any:
        try:
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.detach().item()
                return x.detach().float().mean().item()
            if hasattr(x, "item") and callable(getattr(x, "item")):
                return x.item()
            return x
        except Exception:
            return x

    def _to_number(self, x: Any) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None
