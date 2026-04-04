"""Debugger-aware runtime adjustments for training."""

from __future__ import annotations

import sys
from math import gcd


def maybe_merge_debugger_config(config):
    """Force debugger-safe runtime settings when a debugger is attached."""
    if sys.gettrace() is None:
        return config

    orig_n_envs = int(getattr(config, "n_envs", 1) or 1)
    orig_vectorization_mode = getattr(config, "vectorization_mode", "auto")
    n_steps = int(getattr(config, "n_steps", 1) or 1)
    orig_batch = int(getattr(config, "batch_size", 1) or 1)

    orig_rollout = max(1, int(orig_n_envs) * int(n_steps))
    ratio = float(orig_batch) / float(orig_rollout) if orig_rollout > 0 else 1.0

    config.n_envs = 1
    config.vectorization_mode = "sync"

    new_rollout = max(1, int(config.n_envs) * int(n_steps))
    new_batch = max(1, int(new_rollout * ratio))
    if new_batch > new_rollout:
        new_batch = new_rollout
    if new_rollout % new_batch != 0:
        divisor = gcd(int(new_rollout), int(new_batch))
        new_batch = int(divisor) if int(divisor) > 0 else 1

    old_batch = getattr(config, "batch_size", new_batch)
    config.batch_size = int(new_batch)
    print(
        f"Debugger detected: forcing n_envs=1, vectorization_mode='sync'; "
        f"batch_size {old_batch}→{config.batch_size} "
        f"(was n_envs={orig_n_envs}, vectorization_mode='{orig_vectorization_mode}')."
    )
    return config
