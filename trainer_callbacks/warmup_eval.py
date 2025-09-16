from __future__ import annotations

import pytorch_lightning as pl

class WarmupEvalCallback(pl.Callback):
    def __init__(self, *, warmup_epochs: int, eval_freq_epochs: int):
        self.warmup_epochs = warmup_epochs
        self.eval_freq_epochs = eval_freq_epochs

    def on_fit_start(self, trainer, pl_module):
        # disable val during warmup
        if self.warmup_epochs > 0:
            trainer.limit_val_batches = 0

    def on_train_epoch_end(self, trainer, pl_module):
        # after warmup, enable val and set the cadence
        if trainer.current_epoch + 1 == self.warmup_epochs:
            trainer.limit_val_batches = 1.0
            trainer.check_val_every_n_epoch = self.eval_freq_epochs