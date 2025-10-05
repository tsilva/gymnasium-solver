"""Builder for trainer callbacks."""

from typing import TYPE_CHECKING

from trainer_callbacks import (
    ConsoleSummaryCallback,
    DispatchMetricsCallback,
    EarlyStoppingCallback,
    KeyboardShortcutCallback,
    ModelCheckpointCallback,
    MonitorMetricsCallback,
    PlateauInterventionCallback,
    UploadRunCallback,
    WandbVideoLoggerCallback,
    WarmupEvalCallback,
)
from utils.schedule_resolver import build_hyperparameter_scheduler_callbacks

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class CallbackBuilder:
    """Builds trainer callbacks for an agent."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self.config = agent.config
        self.run = agent.run

    def build(self) -> list:
        """Assemble trainer callbacks."""
        callbacks = []

        # Keyboard shortcuts: enable interactive control during training
        #callbacks.append(KeyboardShortcutCallback())

        # In case eval warmup is active, add a callback to enable validation only after warmup
        if self.config.eval_warmup_epochs > 0:
            callbacks.append(
                WarmupEvalCallback(
                    warmup_epochs=self.config.eval_warmup_epochs,
                    eval_freq_epochs=self.config.eval_freq_epochs,
                )
            )

        # Monitor metrics: add alerts if any are triggered
        # (must be before DispatchMetricsCallback because alerts are reported as metrics)
        callbacks.append(MonitorMetricsCallback())

        # Metrics dispatcher: aggregates epoch metrics and logs to Lightning
        callbacks.append(DispatchMetricsCallback())

        # Auto-wire hyperparameter schedulers: scan config for any *_schedule fields
        scheduler_callbacks = self._build_scheduler_callbacks()
        callbacks.extend(scheduler_callbacks)

        # Plateau intervention: adjust hyperparameters when training stagnates
        plateau_callbacks = self._build_plateau_intervention_callbacks()
        callbacks.extend(plateau_callbacks)

        # Early stopping callbacks: must run BEFORE ModelCheckpointCallback
        # so that trainer.should_stop is set when checkpoint logic runs
        early_stop_callbacks = self._build_early_stopping_callbacks()
        callbacks.extend(early_stop_callbacks)

        # Checkpointing: save best/last models and metrics
        # Must run AFTER early stopping callbacks to detect trainer.should_stop
        callbacks.append(
            ModelCheckpointCallback(
                run=self.run,
                metric="val/roll/ep_rew/mean",
                mode="max",
            )
        )

        # Video logger watches checkpoints directory recursively for videos (only if W&B is enabled)
        if getattr(self.config, "enable_wandb", True):
            callbacks.append(
                WandbVideoLoggerCallback(
                    media_root=self.run.checkpoints_dir,
                    namespace_depth=1,
                )
            )

        # Also print a terminal summary and alerts recap at the end of training
        callbacks.append(ConsoleSummaryCallback())

        # Optionally upload run folder to W&B after training completes
        if getattr(self.config, "enable_wandb", True):
            callbacks.append(UploadRunCallback(run_dir=self.run.run_dir))

        return callbacks

    def _build_scheduler_callbacks(self) -> list:
        """Build hyperparameter scheduler callbacks."""

        def _set_policy_lr(module: "BaseAgent", lr: float) -> None:
            module._change_optimizers_lr(lr)

        set_value_fn_map = {
            "policy_lr": _set_policy_lr,
        }

        return build_hyperparameter_scheduler_callbacks(
            config=self.config,
            module=self.agent,
            set_value_fn_map=set_value_fn_map,
        )

    def _build_plateau_intervention_callbacks(self) -> list:
        """Build plateau intervention callbacks."""
        if not self.config.plateau_interventions:
            return []

        plateau_config = self.config.plateau_interventions

        # Build set_value_fn_map for parameter setters
        def _set_policy_lr(module: "BaseAgent", lr: float) -> None:
            module._change_optimizers_lr(lr)

        set_value_fn_map = {
            "policy_lr": _set_policy_lr,
        }

        return [
            PlateauInterventionCallback(
                monitor=plateau_config["monitor"],
                patience=plateau_config["patience"],
                actions=plateau_config["actions"],
                mode=plateau_config.get("mode", "max"),
                min_delta=plateau_config.get("min_delta", 0.0),
                cooldown=plateau_config.get("cooldown", 0),
                set_value_fn_map=set_value_fn_map,
            )
        ]

    def _build_early_stopping_callbacks(self) -> list:
        """Build early stopping callbacks."""
        callbacks = []

        # Note: max_env_steps early stopping is handled in BaseAgent.on_train_epoch_start()
        # to prevent overshooting the budget by checking before collecting the next rollout

        # If defined in config, early stop when mean training reward reaches a threshold
        # Config value can be True (use env spec threshold) or a float (override threshold)
        if self.config.early_stop_on_train_threshold:
            train_env = self.agent.get_env("train")
            if isinstance(self.config.early_stop_on_train_threshold, float):
                reward_threshold = self.config.early_stop_on_train_threshold
            else:
                reward_threshold = train_env.get_return_threshold()
            callbacks.append(EarlyStoppingCallback("train/roll/ep_rew/mean", reward_threshold))

        # If defined in config, early stop when mean validation reward reaches a threshold
        # Config value can be True (use env spec threshold) or a float (override threshold)
        if self.config.early_stop_on_eval_threshold:
            val_env = self.agent.get_env("val")
            if isinstance(self.config.early_stop_on_eval_threshold, float):
                reward_threshold = self.config.early_stop_on_eval_threshold
            else:
                reward_threshold = val_env.get_return_threshold()
            callbacks.append(EarlyStoppingCallback("val/roll/ep_rew/mean", reward_threshold))

        return callbacks
