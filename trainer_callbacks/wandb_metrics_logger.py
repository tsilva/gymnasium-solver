from __future__ import annotations

import pytorch_lightning as pl

class WandbMetricsLoggerCallback(pl.Callback):

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log_dict(pl_module._last_epoch_metrics)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.log_dict(pl_module._last_epoch_metrics)
