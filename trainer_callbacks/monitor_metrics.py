from __future__ import annotations

from typing import List

import pytorch_lightning as pl

class MonitorMetricsCallback(pl.Callback):

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        alerts = pl_module.metrics_monitor.check(epoch=trainer.current_epoch)
        added_alerts = alerts["added"]
        removed_alerts = alerts["removed"]
        self._record_alerts_as_metrics(pl_module, added_alerts, removed_alerts)

    def _record_alerts_as_metrics(self, pl_module: pl.LightningModule, added_alerts: List[str], removed_alerts: List[str]) -> None:
        add_keys = added_alerts
        remove_keys = removed_alerts
        add_values = [1 for _ in add_keys]
        remove_values = [0 for _ in remove_keys]
        keys = add_keys + remove_keys
        values = add_values + remove_values
        for key, value in zip(keys, values):
            namespace = key.split("/")[0]
            subkey = "/".join(key.split("/")[1:])        
            pl_module.metrics_recorder.record(namespace, {f"alert/{subkey}": value})
