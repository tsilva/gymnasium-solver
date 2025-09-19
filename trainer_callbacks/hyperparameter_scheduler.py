from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from utils.schedulers import resolve as resolve_schedule


# TODO: REFACTOR this file
class HyperparameterScheduler(pl.Callback):
    """Callback that updates scheduled hyperparameters at epoch end.

    Supports:
      - policy_lr (optimizer param groups)
      - clip_range (PPO-specific attribute on the module)
    """

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D401
        cfg = getattr(pl_module, "config", None)
        if cfg is None:
            return

        progress = 0.0
        _calc = getattr(pl_module, "_calc_training_progress", None)
        if callable(_calc):
            try:
                progress = float(_calc())
            except Exception:
                progress = 0.0

        # policy_lr schedule
        if getattr(cfg, "policy_lr_schedule", None):
            sched_fn = resolve_schedule(cfg.policy_lr_schedule)
            if sched_fn is not None and getattr(cfg, "policy_lr", None) is not None:
                new_lr = float(sched_fn(float(cfg.policy_lr), progress))
                self._set_policy_lr(pl_module, new_lr)
                # Log under train namespace (mirrors existing behavior)
                self._record(pl_module, {"policy_lr": new_lr})

        # clip_range schedule (PPO)
        if getattr(cfg, "clip_range_schedule", None):
            sched_fn = resolve_schedule(cfg.clip_range_schedule)
            if sched_fn is not None and getattr(cfg, "clip_range", None) is not None:
                new_clip = float(sched_fn(float(cfg.clip_range), progress))
                # Assign to agent attribute if present
                if hasattr(pl_module, "clip_range"):
                    setattr(pl_module, "clip_range", new_clip)
                self._record(pl_module, {"clip_range": new_clip})

    def _set_policy_lr(self, pl_module: pl.LightningModule, lr: float) -> None:
        optimizers = pl_module.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            for pg in getattr(opt, "param_groups", []):
                pg["lr"] = lr

    def _record(self, pl_module: pl.LightningModule, metrics: dict[str, Any]) -> None:
        # Record scheduled values under the train namespace
        try:
            pl_module.metrics_recorder.record("train", metrics)
        except Exception:
            pass

