"""Torch-related utility helpers.

Contains helpers that operate on torch modules, devices, and inference context.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager

import torch


@contextmanager
def inference_ctx(*modules):
    """
    Temporarily puts all passed nn.Module objects in eval mode and disables grad-tracking.
    Restores their original training flag afterwards.

    Usage:
        with inference_ctx(actor, critic):
            ... collect trajectories ...
    """
    import torch.nn as nn

    # Filter out non-modules and Nones; flatten lists/tuples
    flat = [
        m
        for m in itertools.chain.from_iterable(
            (m if isinstance(m, (list, tuple)) else (m,)) for m in modules
        )
        if isinstance(m, nn.Module) and m is not None
    ]

    was_training = [m.training for m in flat]
    try:
        for m in flat:
            m.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, flag in zip(flat, was_training):
            if flag:  # only restore if it was in train mode
                m.train()


def _device_of(module: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter of a module, or CPU if unknown."""
    if not hasattr(module, "parameters"):
        return torch.device("cpu")
    return next(module.parameters()).device
