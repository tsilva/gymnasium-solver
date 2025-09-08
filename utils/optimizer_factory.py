from __future__ import annotations

import torch


def build_optimizer(*, params, optimizer, lr: float) -> torch.optim.Optimizer:
    """Create and return a torch optimizer for given params.

    - Supports 'adam', 'adamw', and 'sgd' identifiers (str or Enum).
    - Uses the provided learning rate without altering other defaults.
    """
    # Normalize optimizer identifier (supports Enum or string)
    opt_id = getattr(optimizer, "value", optimizer)
    opt_id = str(opt_id).lower()

    optimizer_class = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[opt_id]

    return optimizer_class(params, lr=lr)

