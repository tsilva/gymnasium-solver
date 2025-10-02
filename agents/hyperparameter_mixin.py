"""Mixin for hyperparameter management in agents."""

from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class HyperparameterMixin:
    """Mixin providing hyperparameter management capabilities."""

    def _change_optimizers_lr(self: "BaseAgent", lr: float) -> None:
        """Change learning rate across all optimizers.

        Args:
            lr: New learning rate value
        """
        # Keep attribute in sync for logging/inspection
        self.policy_lr = lr
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr

    def _change_n_epochs(self: "BaseAgent", n_epochs: int) -> None:
        """Change number of epochs for training.

        Args:
            n_epochs: New number of epochs
        """
        self.n_epochs = n_epochs
        self._train_dataloader.sampler.num_passes = n_epochs

    def _read_hyperparameters_from_run(self: "BaseAgent") -> None:
        """Read hyperparameters from run config and apply changes."""
        loaded_config = asdict(self.run.load_config())
        current_config = asdict(self.config)

        # Identify parameters with active schedules (to skip reloading them)
        scheduled_params = set()
        for key in current_config.keys():
            if key.endswith("_schedule") and current_config.get(key):
                param = key[: -len("_schedule")]
                scheduled_params.add(param)

        changes_map = {}
        for key, value in loaded_config.items():
            if type(value) in [list, tuple, dict, None]:
                continue
            # Skip parameters with active schedules
            if key in scheduled_params:
                continue
            current_value = current_config.get(key, None)
            if value != current_value:
                changes_map[key] = value

        if changes_map:
            self.on_hyperparams_change(changes_map)

    def on_hyperparams_change(self: "BaseAgent", changes_map: dict) -> None:
        """Handle hyperparameter changes.

        Args:
            changes_map: Mapping of parameter names to new values
        """
        for key, value in changes_map.items():
            if not hasattr(self.config, key):
                continue
            setattr(self.config, key, value)
            if key == "policy_lr":
                self._change_optimizers_lr(value)
            elif key == "clip_range":
                self.clip_range = value
            elif key == "vf_coef":
                self.vf_coef = value
            elif key == "ent_coef":
                self.ent_coef = value
            elif key == "n_epochs":
                self._change_n_epochs(value)
        print(f"Hyperparameters changed from run: {changes_map}")

    def _log_hyperparameters(self: "BaseAgent") -> None:
        """Log current hyperparameter values."""
        metrics = {
            "n_epochs": self.n_epochs,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "clip_range": self.clip_range,
            "policy_lr": self.policy_lr,
        }
        prefixed = {f"hp/{k}": v for k, v in metrics.items()}
        self.metrics_recorder.record("train", prefixed)

    def set_hyperparameter(self: "BaseAgent", param: str, value: float) -> None:
        """Set a hyperparameter value. Called by HyperparameterSchedulerCallback.

        Args:
            param: Parameter name
            value: New value
        """
        setattr(self, param, value)
        if hasattr(self.config, param):
            setattr(self.config, param, value)
