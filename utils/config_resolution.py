"""Config finalization orchestration.

This module centralizes the order-sensitive resolution pipeline that runs after
dataclass initialization. Keeping the sequence in one place makes config
behavior easier to audit without changing public APIs.
"""

from __future__ import annotations

from typing import Any


def finalize_config(config: Any) -> None:
    """Resolve, normalize, and validate a config instance in legacy order."""
    config._resolve_defaults()
    config._resolve_n_envs()
    config._resolve_atari_defaults()
    config._resolve_vizdoom_defaults()
    config._resolve_retro_defaults()
    config._resolve_numeric_strings()
    config._resolve_batch_size()
    config._resolve_eval_warmup_epochs()
    config._resolve_schedules()
    config._resolve_schedule_defaults()
    config._resolve_policy()
    config.validate()
