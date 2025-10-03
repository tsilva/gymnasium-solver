from __future__ import annotations

from typing import List

import pytorch_lightning as pl


def build_trainer(
    config,
    *,
    logger: List[pl.loggers.Logger],
    callbacks: List[pl.callbacks.Callback],
) -> pl.Trainer:
    # When number of devices is not specified, default to 1
    if config.devices is None: config.devices = 1

    # If warmup is active, request validation every epoch and gate in hooks
    eval_freq_epochs = 1 if (config.eval_freq_epochs is not None and config.eval_warmup_epochs > 0) else config.eval_freq_epochs
    limit_val_batches = 0 if eval_freq_epochs is None else 1.0
    check_val_every_n_epoch = eval_freq_epochs if eval_freq_epochs is not None else 1

    # Calculate log_every_n_steps based on batches per epoch
    batches_per_epoch = (config.n_envs * config.n_steps) // config.batch_size
    log_every_n_steps = max(1, batches_per_epoch // 10)  # Log ~10 times per epoch

    return pl.Trainer(
        logger=logger,
        max_epochs=config.max_epochs if config.max_epochs is not None else -1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator=config.accelerator,
        devices=config.devices,
        reload_dataloaders_every_n_epochs=0,
        val_check_interval=None,
        check_val_every_n_epoch=check_val_every_n_epoch,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks
    )
