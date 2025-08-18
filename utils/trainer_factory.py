from __future__ import annotations

import warnings
import pytorch_lightning as pl

# Silence PL's advisory about low num_workers on the validation DataLoader.
# Our validation loop uses environment rollouts; the val_dataloader is a dummy.
try:  # PL <= 2.x (pytorch_lightning namespace)
    from pytorch_lightning.utilities.warnings import PossibleUserWarning as _PLPossibleUserWarning  # type: ignore
except Exception:  # pragma: no cover - fallback for different versions
    try:  # PL >= 2.x (lightning.pytorch namespace)
        from lightning.pytorch.utilities.warnings import PossibleUserWarning as _PLPossibleUserWarning  # type: ignore
    except Exception:  # pragma: no cover
        _PLPossibleUserWarning = None  # type: ignore

# Apply a targeted filter that matches only the val_dataloader workers hint
_warning_categories = [c for c in (_PLPossibleUserWarning, UserWarning, Warning) if c is not None]
for _cat in _warning_categories:
    warnings.filterwarnings(
        "ignore",
        message=r".*'val_dataloader' does not have many workers.*",
        category=_cat,
    )


def build_trainer(*, logger, callbacks, validation_controls, max_epochs, accelerator="cpu", devices=None) -> pl.Trainer:
    """Small wrapper that centralizes our Trainer construction defaults.

    Keeping this here reduces BaseAgent bloat and makes defaults easy to reuse.
    """
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
