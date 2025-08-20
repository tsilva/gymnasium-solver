"""
Test-only lightweight stub of `pytorch_lightning` to allow unit tests to import
and run without the heavy dependency. Provides the minimal surface used by the
codebase: LightningModule API, a no-op Trainer type, and a WandbLogger shim
exposed under pytorch_lightning.loggers.

This file lives under tests/ so it doesn't pollute the main codebase.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import torch


class LightningModule(torch.nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - match torch signature
        super().__init__()
        # Lightning uses this to control optimization; default False in our project
        self.automatic_optimization: bool = False
        # Last logged dict for tests that may inspect it (optional)
        self._last_log_dict: Dict[str, Any] = {}

    # --- Minimal APIs used by BaseAgent/PPO/tests ---
    def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - no-op stub
        return None

    def log_dict(self, d: Dict[str, Any]) -> None:
        # Store and ignore; real Lightning forwards to logger
        self._last_log_dict = dict(d)

    def log(self, key: str, value: Any) -> None:  # pragma: no cover - rarely used in tests
        self._last_log_dict = {key: value}

    def manual_backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    # In real PL, Trainer wires optimizers; here lazily create if missing
    def optimizers(self) -> List[torch.optim.Optimizer] | torch.optim.Optimizer:
        opt = getattr(self, "_optimizers", None)
        if opt is None:
            opt = self.configure_optimizers() if hasattr(self, "configure_optimizers") else None
            if opt is None:
                raise RuntimeError("No optimizers available; override configure_optimizers or set self._optimizers")
            self._optimizers = opt  # type: ignore[assignment]
        return opt


class Trainer:  # pragma: no cover - type placeholder for annotations
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.logger = SimpleNamespace(log_metrics=lambda *a, **k: None)
        self.logged_metrics: Dict[str, Any] = {}
        self.callback_metrics: Dict[str, Any] = {}
        self.progress_bar_metrics: Dict[str, Any] = {}
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.should_stop: bool = False

    def fit(self, *_: Any, **__: Any) -> None:
        # Not used; tests provide their own trainer stub.
        return None


# Expose a tiny WandbLogger under the loggers namespace
class _WandbLogger:
    def __init__(self, project: str, name: str, log_model: bool, config: Dict[str, Any]) -> None:  # noqa: D401
        self.project = project
        self.name = name
        self.log_model = log_model
        self.config = dict(config)
        self.experiment = SimpleNamespace(id="test", define_metric=lambda *a, **k: None)


class loggers:  # noqa: N801 - mimic package for `from pytorch_lightning.loggers import WandbLogger`
    WandbLogger = _WandbLogger
