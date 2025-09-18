# Pytest configuration and lightweight dependency shims for test-time only.
#
# We keep shims here (next to tests) to avoid polluting the main codebase.

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

import torch

# --- Minimal pytorch_lightning shim ---------------------------------------------------------
if "pytorch_lightning" not in sys.modules:
    pl = types.ModuleType("pytorch_lightning")

    class LightningModule(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.automatic_optimization = False
            self._last_log_dict: Dict[str, Any] = {}

        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            return None

        def log_dict(self, d: Dict[str, Any]) -> None:
            self._last_log_dict = dict(d)

        def log(self, key: str, value: Any) -> None:
            self._last_log_dict = {key: value}

        def manual_backward(self, loss: torch.Tensor) -> None:
            loss.backward()

        def optimizers(self) -> List[torch.optim.Optimizer] | torch.optim.Optimizer:
            opt = getattr(self, "_optimizers", None)
            if opt is None:
                opt = self.configure_optimizers() if hasattr(self, "configure_optimizers") else None
                if opt is None:
                    raise RuntimeError(
                        "No optimizers available; override configure_optimizers or set self._optimizers"
                    )
                self._optimizers = opt
            return opt

    class Trainer:  # pragma: no cover - placeholder for annotations
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.logger = SimpleNamespace(log_metrics=lambda *a, **k: None)
            self.logged_metrics: Dict[str, Any] = {}
            self.callback_metrics: Dict[str, Any] = {}
            self.progress_bar_metrics: Dict[str, Any] = {}
            self.current_epoch: int = 0
            self.global_step: int = 0
            self.should_stop: bool = False

        def fit(self, *_: Any, **__: Any) -> None:
            return None

    # Expose loggers.WandbLogger
    class _WandbLogger:
        def __init__(self, project: str, name: str, log_model: bool, config: Dict[str, Any]) -> None:  # noqa: D401
            self.project = project
            self.name = name
            self.log_model = log_model
            self.config = dict(config)
            self.experiment = SimpleNamespace(id="test", define_metric=lambda *a, **k: None)

    loggers = types.SimpleNamespace(WandbLogger=_WandbLogger)

    # Attach to module and register
    pl.LightningModule = LightningModule
    pl.Trainer = Trainer
    pl.loggers = loggers

    sys.modules["pytorch_lightning"] = pl

# -------------------------------------------------------------------------------------------
