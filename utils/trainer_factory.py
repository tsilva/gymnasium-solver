from __future__ import annotations

import pytorch_lightning as pl

def build_trainer(*, logger, callbacks, validation_controls, max_epochs, accelerator="cpu", devices=None) -> pl.Trainer:
    # Lightning requires an explicit positive int for CPU devices; coerce sensible default
    eff_devices = devices
    if (accelerator == "cpu"):
        if eff_devices is None or eff_devices == "auto":
            eff_devices = 1
        elif isinstance(eff_devices, int) and eff_devices <= 0:
            eff_devices = 1

    return pl.Trainer(
        logger=logger,
        max_epochs=max_epochs if max_epochs is not None else -1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=eff_devices,
        reload_dataloaders_every_n_epochs=0,
        val_check_interval=None,
        check_val_every_n_epoch=validation_controls["check_val_every_n_epoch"],
        limit_val_batches=validation_controls["limit_val_batches"],
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
