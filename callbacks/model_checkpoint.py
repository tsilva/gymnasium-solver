"""Compatibility shim mapping callbacks.model_checkpoint to trainer_callbacks.model_checkpoint.

This file exists to satisfy tests and downstream code that import
`callbacks.model_checkpoint` while the actual implementation lives in
`trainer_callbacks.model_checkpoint`.
"""

from trainer_callbacks.model_checkpoint import *  # noqa: F401,F403


