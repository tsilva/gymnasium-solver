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

    # TODO: softcode this
    # NOTE: still not sure if this helps or not
    # NOTE: magic eps from openai baselines: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py#L100
    #extra_kwargs = {
    #    "eps": 1e-5,
    #} if opt_id == "adam" else {}

    return optimizer_class(params, lr=lr)#, **extra_kwargs)

