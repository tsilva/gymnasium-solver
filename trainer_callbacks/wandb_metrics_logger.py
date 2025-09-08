from __future__ import annotations

import pytorch_lightning as pl

class WandbMetricsLoggerCallback(pl.Callback):

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = pl_module._epoch_metrics_buffer.means()
        pl_module.log_dict(metrics)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = pl_module._epoch_metrics_buffer.means()
        pl_module.log_dict(metrics)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = pl_module._epoch_metrics_buffer.means()
        pl_module.log_dict(metrics)
