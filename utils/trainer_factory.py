from __future__ import annotations

import pytorch_lightning as pl

def build_trainer(
    *, 
    logger, 
    callbacks, 
    max_epochs, 
    accelerator="cpu", 
    devices=None,
    eval_freq_epochs=None,
    eval_warmup_epochs=0
) -> pl.Trainer:
    # When number of devices is not specified, set to 1 for CPU
    if devices is None and accelerator == "cpu": devices = 1

    # If warmup is active, request validation every epoch and gate in hooks
    eval_freq_epochs = 1 if (eval_freq_epochs is not None and eval_warmup_epochs > 0) else eval_freq_epochs
    limit_val_batches = 0 if eval_freq_epochs is None else 1.0
    check_val_every_n_epoch = eval_freq_epochs if eval_freq_epochs is not None else 1

    return pl.Trainer(
        logger=logger,
        max_epochs=max_epochs if max_epochs is not None else -1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=devices,
        reload_dataloaders_every_n_epochs=0,
        val_check_interval=None,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
