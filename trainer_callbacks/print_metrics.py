"""Print metrics callback for displaying training progress in a formatted table."""

import re
from typing import Any, Dict, Iterable, List, Optional

# Optional dependency shim for pytorch_lightning.Callback
try:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    BaseCallback = getattr(pl, "Callback", object)
    TrainerType = getattr(pl, "Trainer", object)
    LightningModuleType = getattr(pl, "LightningModule", object)
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    BaseCallback = object
    class TrainerType:  # minimal placeholders for type annotations
        pass
    class LightningModuleType:
        pass

import torch


class PrintMetricsCallback(BaseCallback):
    """
    Periodically prints a table to stdout with the *latest value* for each logged metric.

    Sources:
      - trainer.logged_metrics        (step-level latest values)
      - trainer.callback_metrics      (epoch-level aggregated values)
      - trainer.progress_bar_metrics  (if present; UI-focused values)

    Usage:
      printer = PrintMetricsCallback(every_n_steps=200, every_n_epochs=1, include=[r'^train/', r'^val/'])
      trainer = pl.Trainer(callbacks=[printer], ...)
    """

    def __init__(
        self,
        every_n_steps: Optional[int] = None,   # if None, don't print by step
        every_n_epochs: Optional[int] = 1,     # print each epoch by default
        include: Optional[Iterable[str]] = None,  # regex patterns to keep
        exclude: Optional[Iterable[str]] = None,  # regex patterns to drop
        digits: int = 4,                          # rounding for floats
        metric_precision: Optional[Dict[str, int]] = None,  # precision per metric
        metric_delta_rules: Optional[Dict[str, callable]] = None,  # delta validation rules per metric
        algorithm_metric_rules: Optional[Dict[str, dict]] = None,  # algorithm-specific warning rules
        min_val_width: int = 15,                  # minimum width for values column
        key_priority: Optional[List[str]] = None,  # priority order for sorting keys
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.include = [re.compile(p) for p in (include or [])]
        self.exclude = [re.compile(p) for p in (exclude or [])]
        self.digits = digits
        self.metric_precision = metric_precision or {}
        self.metric_delta_rules = metric_delta_rules or {}
        self.algorithm_metric_rules = algorithm_metric_rules or {}
        self.min_val_width = min_val_width
        # Soft tolerance for minor non-monotonic blips on wall-clock metrics (seconds)
        # This prevents spurious failures when values are sourced/rounded slightly differently
        # between step and epoch prints. Applied to metrics whose key ends with 'time_elapsed'.
        self._delta_soft_tolerance_sec = 1.0
        # Load key priority from config unless explicitly provided
        if key_priority is not None:
            self.key_priority = key_priority
        else:
            try:
                from utils.metrics import get_key_priority
                cfg_key_priority = get_key_priority()
            except Exception:
                cfg_key_priority = None
            # Fallback to previous hardcoded defaults if config not present
            self.key_priority = cfg_key_priority or [
                "train/ep_rew_mean",
                "train/ep_len_mean",
                "train/epoch",
                "train/total_timesteps",
                "train/total_episodes",
                "train/total_rollouts",
                "train/rollout_timesteps",
                "train/rollout_episodes",
                "train/epoch_fps",
                "train/rollout_fps",
                "train/loss",
                "train/policy_loss",
                "train/value_loss",
                "train/entropy_loss",
                "eval/ep_rew_mean",
                "eval/ep_len_mean",
                "eval/epoch",
                "eval/total_timesteps",
                "eval/total_episodes",
                "eval/total_rollouts",
                "eval/rollout_timesteps",
                "eval/rollout_episodes",
                "eval/epoch_fps",
                "eval/rollout_fps",
            ]
        self.previous_metrics = {}  # Store previous values for delta validation
        self._last_printed_metrics = None  # For change detection
        self._change_tol = 1e-12

        # Dedicated table printer to preserve state across prints (avoids global resets)
        from utils.table_printer import NamespaceTablePrinter
        self._printer = NamespaceTablePrinter(
            # Keep numbers compact and colored like before
            compact_numbers=True,
            color=True,
            # In-place update to avoid repeated full prints scrolling the log
            use_ansi_inplace=False,
            # Respect configured value formatting and layout
            metric_precision=self.metric_precision,
            min_val_width=self.min_val_width,
            key_priority=self.key_priority,
        )

    # ---------- hooks ----------
    def on_train_batch_end(self, trainer: TrainerType, pl_module: LightningModuleType, outputs, batch, batch_idx):
        if self.every_n_steps is None:
            return
        # global_step increments after optimizer step; print when divisible
        if trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            self._maybe_print(trainer, stage="train-step")

    def on_validation_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleType):
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_print(trainer, stage="val-epoch")

    def on_train_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleType):
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_print(trainer, stage="train-epoch")

    def on_test_epoch_end(self, trainer: TrainerType, pl_module: LightningModuleType):
        # Always print at test end if enabled by epoch cadence
        if self.every_n_epochs is not None:
            self._maybe_print(trainer, stage="test-epoch")

    # ---------- internals ----------
    def _maybe_print(self, trainer: TrainerType, stage: str):
        # Only the main process prints
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        metrics = self._collect_metrics(trainer)
        metrics = self._filter_metrics(metrics)
        if not metrics:
            return

        # Skip if nothing changed materially since last print
        if not self._metrics_changed(metrics):
            return

        # Validate metric delta rules before printing
        self._validate_metric_deltas(metrics)

        # Check algorithm-specific metric rules and log warnings
        self._check_algorithm_metric_rules(metrics)

        step = getattr(trainer, "global_step", None)
        epoch = getattr(trainer, "current_epoch", None)
        header = f"[{stage}] epoch={epoch} step={step}"
        self._print_table(metrics, header)

        # After printing the table, append the W&B run URL for epoch-end prints
        try:
            if isinstance(stage, str) and stage.endswith("epoch"):
                url = self._get_wandb_run_url(trainer)
                if url:
                    print(self._format_wandb_url_line(url))
        except Exception:
            # Never break training output due to URL printing issues
            pass

        # Update previous metrics for next comparison
        self.previous_metrics.update(metrics)
        self._last_printed_metrics = dict(metrics)

    def _validate_metric_deltas(self, current_metrics: Dict[str, Any]) -> None:
        """Validate that metric deltas follow specified rules."""
        for metric_name, rule_lambda in self.metric_delta_rules.items():
            if metric_name in current_metrics and metric_name in self.previous_metrics:
                current_value = self._to_python_scalar(current_metrics[metric_name])
                previous_value = self._to_python_scalar(self.previous_metrics[metric_name])
                
                # Skip validation if either value is not a number
                if not (self._is_number(current_value) and self._is_number(previous_value)):
                    continue

                # Allow small backward drift for time_elapsed-like metrics to handle
                # clock jitter or differing sources (e.g., step vs epoch timers)
                try:
                    if str(metric_name).endswith("time_elapsed"):
                        if float(previous_value) - float(current_value) <= self._delta_soft_tolerance_sec:
                            # Treat as valid within tolerance
                            continue
                except Exception:
                    # Fall back to strict rule if casting fails
                    pass
                
                try:
                    # Call the lambda with (previous, current) values
                    rule_satisfied = rule_lambda(previous_value, current_value)
                    if not rule_satisfied:
                        raise ValueError(
                            f"Metric delta rule violation for '{metric_name}': "
                            f"previous={previous_value}, current={current_value}. "
                            f"Rule: {rule_lambda.__name__ if hasattr(rule_lambda, '__name__') else 'lambda'}"
                        )
                except Exception as e:
                    if isinstance(e, ValueError) and "Metric delta rule violation" in str(e):
                        raise  # Re-raise our validation errors
                    # For other exceptions (e.g., in lambda evaluation), wrap them
                    raise ValueError(
                        f"Error evaluating metric delta rule for '{metric_name}': {str(e)}"
                    ) from e

    def _check_algorithm_metric_rules(self, current_metrics: Dict[str, Any]) -> None:
        """Check algorithm-specific metric rules and log warnings when violated."""
        
        for metric_name, rule_config in self.algorithm_metric_rules.items():
            if metric_name in current_metrics:
                current_value = self._to_python_scalar(current_metrics[metric_name])
                
                # Skip validation if value is not a number
                if not self._is_number(current_value):
                    continue
                
                try:
                    # Get rule configuration
                    check_func = rule_config.get('check')
                    message_template = rule_config.get('message', f"Algorithm metric rule violated for '{metric_name}'")
                    level = rule_config.get('level', 'warning')
                    
                    if check_func is None:
                        continue
                    
                    # Check if rule is satisfied (different types of checks)
                    rule_satisfied = False
                    
                    if callable(check_func):
                        # For threshold checks that only need current value
                        if check_func.__code__.co_argcount == 1:
                            rule_satisfied = check_func(current_value)
                        # For delta checks that need previous and current values
                        elif metric_name in self.previous_metrics:
                            previous_value = self._to_python_scalar(self.previous_metrics[metric_name])
                            if self._is_number(previous_value):
                                rule_satisfied = check_func(previous_value, current_value)
                            else:
                                continue  # Skip if previous value isn't a number
                        else:
                            continue  # Skip if no previous value for delta check
                    
                    if not rule_satisfied:
                        # Format the warning message
                        if metric_name in self.previous_metrics:
                            previous_value = self._to_python_scalar(self.previous_metrics[metric_name])
                            formatted_message = message_template.format(
                                metric_name=metric_name,
                                current_value=current_value,
                                previous_value=previous_value if self._is_number(previous_value) else 'N/A'
                            )
                        else:
                            formatted_message = message_template.format(
                                metric_name=metric_name,
                                current_value=current_value,
                                previous_value='N/A'
                            )
                        
                        # Log warning or error based on level
                        if level == 'error':
                            print(f"🚨 ALGORITHM ERROR: {formatted_message}")
                        else:
                            print(f"⚠️  ALGORITHM WARNING: {formatted_message}")
                            
                except Exception as e:
                    print(f"⚠️  Error checking algorithm metric rule for '{metric_name}': {str(e)}")

    def _collect_metrics(self, trainer: TrainerType) -> Dict[str, Any]:
        combo: Dict[str, Any] = {}

        # Merge, with later sources taking precedence
        dicts = []
        if hasattr(trainer, "logged_metrics"):
            dicts.append(trainer.logged_metrics)
        if hasattr(trainer, "callback_metrics"):
            dicts.append(trainer.callback_metrics)
        if hasattr(trainer, "progress_bar_metrics"):  # not always present
            dicts.append(trainer.progress_bar_metrics)

        for d in dicts:
            for k, v in d.items():
                combo[k] = self._to_python_scalar(v)

        # Opportunistically refresh canonical step metric from the module's collector
        try:
            pl_module = getattr(trainer, "lightning_module", None)
            tc = getattr(pl_module, "train_collector", None) if pl_module is not None else None
            if tc is not None and hasattr(tc, "total_steps"):
                combo["train/total_timesteps"] = int(getattr(tc, "total_steps", 0))
        except Exception:
            # Never break printing due to instrumentation
            pass

        # Drop common housekeeping keys if present
        for k in ["epoch", "step", "global_step"]:
            combo.pop(k, None)
        return combo

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self.include:
            metrics = {k: v for k, v in metrics.items() if any(p.search(k) for p in self.include)}
        if self.exclude:
            metrics = {k: v for k, v in metrics.items() if not any(p.search(k) for p in self.exclude)}
        return metrics

    def _to_python_scalar(self, x: Any) -> Any:
        # Convert tensors and numpy scalars to plain Python types when possible
        try:
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.detach().item()
                return x.detach().float().mean().item()
            # numpy scalar support without importing numpy explicitly
            if hasattr(x, "item") and callable(getattr(x, "item")):
                return x.item()
            return x
        except Exception:
            return x

    def _print_table(self, metrics: Dict[str, Any], header: str):
        # Use the dedicated printer instance so deltas compare with previous call reliably
        # Header remains available for future use if we want to render it in the table
        _ = header  # currently unused in rendering
        self._printer.update(metrics)

    def _get_wandb_run_url(self, trainer: TrainerType) -> Optional[str]:
        """Return the W&B run URL if available from the trainer's logger."""
        try:
            logger = getattr(trainer, "logger", None)
            if logger is None:
                return None

            # Single logger case (WandbLogger)
            exp = getattr(logger, "experiment", None)
            if exp is not None:
                url = getattr(exp, "url", None)
                return str(url) if url else None

            # Logger collection case
            loggers = getattr(logger, "loggers", None)
            if isinstance(loggers, (list, tuple)):
                for lg in loggers:
                    exp = getattr(lg, "experiment", None)
                    if exp is not None:
                        url = getattr(exp, "url", None)
                        if url:
                            return str(url)
        except Exception:
            return None
        return None

    def _format_wandb_url_line(self, url: str) -> str:
        """Format a terminal-friendly hyperlinked line for the W&B run URL.

        Uses OSC 8 hyperlinks when supported; also includes the raw URL.
        """
        try:
            esc = "\x1b"
            link = f"{esc}]8;;{url}{esc}\\Open W&B run{esc}]8;;{esc}\\"
            return f"W&B: {link}  ({url})"
        except Exception:
            return f"W&B: {url}"

    def _metrics_changed(self, metrics: Dict[str, Any]) -> bool:
        """Return True if any metric changed beyond tolerance or new keys appeared."""
        prev = self._last_printed_metrics
        if prev is None:
            return True
        # If key sets differ, consider it changed
        if set(metrics.keys()) != set(prev.keys()):
            return True
        for k, v in metrics.items():
            pv = prev.get(k)
            if self._is_number(v) and self._is_number(pv):
                if abs(float(v) - float(pv)) > self._change_tol:
                    return True
            else:
                if v != pv:
                    return True
        return False

    def _format_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.digits}f}"
        return str(v)

    def _is_number(self, x: Any) -> bool:
        import numbers
        return isinstance(x, numbers.Number)
