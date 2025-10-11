"""Plateau intervention callback for hyperparameter adjustment when training stagnates."""

from collections import deque
from typing import Any, Callable, Dict, List, Optional

import pytorch_lightning as pl


class PlateauInterventionCallback(pl.Callback):
    """Adjust hyperparameters when a monitored metric plateaus.

    When a metric (e.g., train/roll/ep_rew/mean) stops improving for `patience` epochs,
    this callback cycles through a list of interventions (e.g., reduce LR, increase entropy).

    Supports:
    - Multiple interventions applied sequentially on repeated plateaus
    - Cooldown period after each intervention
    - Optional revert if intervention makes metric worse
    - Min/max bounds on parameter values

    Example:
        callback = PlateauInterventionCallback(
            monitor="train/roll/ep_rew/mean",
            patience=20,
            actions=[
                {"param": "policy_lr", "operation": "multiply", "factor": 0.5, "min": 1e-6},
                {"param": "ent_coef", "operation": "multiply", "factor": 2.0, "max": 0.1},
            ],
            mode="max",
            cooldown=10,
        )
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        actions: List[Dict[str, Any]],
        mode: str = "max",
        min_delta: float = 0.0,
        cooldown: int = 0,
        set_value_fn_map: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize plateau intervention callback.

        Args:
            monitor: Metric to monitor (e.g., "train/roll/ep_rew/mean")
            patience: Number of epochs without improvement before intervention
            actions: List of intervention dicts with keys:
                - param: Parameter name (e.g., "policy_lr")
                - operation: "multiply", "add", or "set"
                - factor/value: Amount to adjust (for multiply/add) or new value (for set)
                - min: Optional minimum bound
                - max: Optional maximum bound
                - revert_on_worse: Optional bool to revert if metric worsens (default: True)
            mode: "max" to maximize metric, "min" to minimize
            min_delta: Minimum change to count as improvement
            cooldown: Epochs to wait after intervention before detecting new plateau
            set_value_fn_map: Optional dict mapping param names to custom setter functions
        """
        super().__init__()

        assert mode in ("max", "min"), f"mode must be 'max' or 'min', got {mode}"
        assert patience > 0, "patience must be positive"
        assert cooldown >= 0, "cooldown must be non-negative"
        assert len(actions) > 0, "actions list cannot be empty"

        self.monitor = monitor
        self.patience = patience
        self.actions = actions
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.set_value_fn_map = set_value_fn_map or {}

        # Parse actions and set defaults
        for action in self.actions:
            action.setdefault("revert_on_worse", True)
            assert action["operation"] in ("multiply", "add", "set"), \
                f"operation must be 'multiply', 'add', or 'set', got {action['operation']}"

        # State tracking
        self.wait_count = 0
        self.cooldown_count = 0
        self.best_value = None
        self.current_action_idx = 0
        self.last_intervention_epoch = -1
        self.intervention_count = 0

        # History for revert
        self.param_snapshot = {}
        self.metric_before_intervention = None

        # Track recent metric history for plateau detection
        self.metric_history = deque(maxlen=patience + 1)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Get current metric value from recorder
        current_value = self._get_metric_value(pl_module)
        if current_value is None:
            return

        # Update metric history
        self.metric_history.append(current_value)

        # Handle cooldown period after intervention
        if self.cooldown_count > 0:
            self.cooldown_count -= 1

            # Check if we should revert the last intervention
            if self.cooldown_count == 0:
                self._check_revert(pl_module, current_value)

            return

        # Update best value
        if self.best_value is None:
            self.best_value = current_value
            return

        # Check if metric improved
        improved = self._is_improved(current_value, self.best_value)

        if improved:
            self.best_value = current_value
            self.wait_count = 0
        else:
            self.wait_count += 1

        # Check if plateau detected
        if self.wait_count >= self.patience:
            self._apply_intervention(trainer, pl_module, current_value)
            self.wait_count = 0

    def _get_metric_value(self, pl_module: pl.LightningModule) -> Optional[float]:
        """Extract metric value from recorder."""
        try:
            recorder = pl_module.metrics_recorder
            history_dict = recorder.history()  # Call method, not access property
            metric_history = history_dict.get(self.monitor)
            if metric_history is None or len(metric_history) == 0:
                return None
            # Get the most recent value (history is list of (step, value) tuples)
            return float(metric_history[-1][1])
        except (AttributeError, KeyError, IndexError, TypeError):
            return None

    def _is_improved(self, current: float, best: float) -> bool:
        """Check if current value is an improvement over best."""
        if self.mode == "max":
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta

    def _apply_intervention(self, trainer: pl.Trainer, pl_module: pl.LightningModule, current_value: float) -> None:
        """Apply the next intervention in the cycle."""
        action = self.actions[self.current_action_idx]
        param_name = action["param"]

        # Take snapshot for potential revert
        current_param_value = getattr(pl_module, param_name, None)
        if current_param_value is None:
            print(f"Warning: Parameter '{param_name}' not found on module, skipping intervention")
            return

        self.param_snapshot[param_name] = float(current_param_value)
        self.metric_before_intervention = current_value

        # Compute new value
        operation = action["operation"]
        if operation == "multiply":
            new_value = current_param_value * action["factor"]
        elif operation == "add":
            new_value = current_param_value + action["value"]
        elif operation == "set":
            new_value = action["value"]

        # Apply bounds
        if "min" in action:
            new_value = max(new_value, action["min"])
        if "max" in action:
            new_value = min(new_value, action["max"])

        # Set the new value
        self._set_parameter_value(pl_module, param_name, new_value)

        # Log the intervention
        self.intervention_count += 1
        self.last_intervention_epoch = int(pl_module.current_epoch)

        print(f"\n{'='*70}")
        print(f"PLATEAU DETECTED: {self.monitor} stagnant for {self.patience} epochs")
        print(f"Intervention #{self.intervention_count}: {param_name}")
        print(f"  {current_param_value:.6g} → {new_value:.6g} ({operation} by {action.get('factor', action.get('value'))})")
        print(f"  Cooldown: {self.cooldown} epochs")
        print(f"{'='*70}\n")

        # Log as metrics
        pl_module.metrics_recorder.record("train", {
            "plateau/intervention_count": self.intervention_count,
            "plateau/intervention_epoch": self.last_intervention_epoch,
            f"plateau/{param_name}_old": current_param_value,
            f"plateau/{param_name}_new": new_value,
        })

        # Start cooldown period
        self.cooldown_count = self.cooldown

        # Advance to next action in cycle
        self.current_action_idx = (self.current_action_idx + 1) % len(self.actions)

    def _check_revert(self, pl_module: pl.LightningModule, current_value: float) -> None:
        """Check if we should revert the last intervention."""
        action = self.actions[(self.current_action_idx - 1) % len(self.actions)]

        # Only revert if configured to do so
        if not action.get("revert_on_worse", True):
            return

        param_name = action["param"]

        # Check if metric got worse
        if self.metric_before_intervention is not None:
            got_worse = not self._is_improved(current_value, self.metric_before_intervention)

            if got_worse and param_name in self.param_snapshot:
                old_value = self.param_snapshot[param_name]
                current_param_value = getattr(pl_module, param_name)

                # Revert the change
                self._set_parameter_value(pl_module, param_name, old_value)

                print(f"\n{'='*70}")
                print(f"REVERTING INTERVENTION: {self.monitor} worsened")
                print(f"  {param_name}: {current_param_value:.6g} → {old_value:.6g} (reverted)")
                print(f"{'='*70}\n")

                # Log revert
                pl_module.metrics_recorder.record("train", {
                    "plateau/reverted": 1,
                    f"plateau/{param_name}_reverted_to": old_value,
                })

    def _set_parameter_value(self, pl_module: pl.LightningModule, param_name: str, value: float) -> None:
        """Set parameter value using custom setter if available."""
        if param_name in self.set_value_fn_map:
            self.set_value_fn_map[param_name](pl_module, value)
        else:
            # Use standard hyperparameter setter
            pl_module.set_hyperparameter(param_name, value)
