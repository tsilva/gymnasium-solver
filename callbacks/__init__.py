"""Compatibility package re-exporting trainer callbacks under legacy name.

This module allows tests and external code that import from `callbacks.*`
to continue working by forwarding imports to `trainer_callbacks.*`.
"""

from .model_checkpoint import ModelCheckpointCallback  # noqa: F401


