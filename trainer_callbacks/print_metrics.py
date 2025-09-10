"""Print metrics callback for displaying training progress in a formatted table."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

import pytorch_lightning as pl

import torch


class PrintMetricsCallback(pl.Callback):
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
        every_n_steps: int | None = None,   # if None, don't print by step
        every_n_epochs: int | None = 1,     # print each epoch by default
        include: Iterable[str] | None = None,  # regex patterns to keep
        exclude: Iterable[str] | None = None,  # regex patterns to drop
        digits: int = 4,                          # rounding for floats
        metric_precision: Dict[str, int] | None = None,  # precision per metric
        metric_delta_rules: Dict[str, callable] | None = None,  # delta validation rules per metric
        algorithm_metric_rules: Dict[str, dict] | None = None,  # algorithm-specific warning rules
        min_val_width: int = 15,                  # minimum width for values column
        key_priority: List[str] | None = None,  # priority order for sorting keys
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
        self.key_priority = key_priority
        self.previous_metrics = {}  # Store previous values for delta validation
        self._last_printed_metrics = None  # For change detection
        self._change_tol = 1e-12

        # Dedicated table printer to preserve state across prints (avoids global resets)
        from utils.table_printer import NamespaceTablePrinter
        
        # Load highlight configuration from metrics.yaml if available
        from utils.metrics import get_highlight_config, get_metric_bounds
        _hl_cfg = get_highlight_config()
        _hl_value_bold_for = _hl_cfg.get('value_bold_metrics', set())
        _hl_row_for = _hl_cfg.get('row_metrics', set())
        _hl_row_bg_color = _hl_cfg.get('row_bg_color', 'bg_blue')
        _hl_row_bold = bool(_hl_cfg.get('row_bold', True))
        _metric_bounds = get_metric_bounds()
        self._metric_bounds = dict(_metric_bounds)


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
            highlight_value_bold_for=_hl_value_bold_for,
            highlight_row_for=_hl_row_for,
            highlight_row_bg_color=_hl_row_bg_color,
            highlight_row_bold=_hl_row_bold,
            # Highlight out-of-range metrics in yellow per metrics.yaml bounds
            metric_bounds=_metric_bounds,
            highlight_bounds_bg_color='bg_yellow',
        )

    # ---------- hooks ----------
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if self.every_n_steps is None:
            return
        # global_step increments after optimizer step; print when divisible
        if trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            self._maybe_print(trainer, stage="train-step")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_print(trainer, stage="val-epoch")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_print(trainer, stage="train-epoch")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Always print at test end if enabled by epoch cadence
        if self.every_n_epochs is not None:
            self._maybe_print(trainer, stage="test-epoch")

    # ---------- internals ----------
    def _maybe_print(self, trainer: pl.Trainer, stage: str):
        # Only the main process prints
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        metrics = self._collect_metrics(trainer)
        metrics = self._filter_metrics(metrics)
        if not metrics:
            return

        # Skip if nothing changed materially since last print, unless we are about to stop.
        # When early stopping triggers, force a final print of the most recent metrics table
        # so the last epoch is visible in the console output.
        try:
            should_force = bool(getattr(trainer, "should_stop", False))
        except Exception:
            should_force = False
        if not should_force:
            if not self._metrics_changed(metrics):
                return

        # Validate metric delta rules before printing
        self._validate_metric_deltas(metrics)

        # Check algorithm-specific metric rules and log warnings
        self._check_algorithm_metric_rules(metrics)

        step = getattr(trainer, "global_step", None)
        epoch = getattr(trainer, "current_epoch", None)
        header = f"[{stage}] epoch={epoch} step={step}"

        # Print the W&B run URL above the table (epoch-end prints only)
        try:
            if isinstance(stage, str) and stage.endswith("epoch"):
                url = self._get_wandb_run_url(trainer)
                if url:
                    print(self._format_wandb_url_line(url))
        except Exception:
            # Never break training output due to URL printing issues
            pass

        self._print_table(metrics, header)

        # Update previous metrics for next comparison
        self.previous_metrics.update(metrics)
        self._last_printed_metrics = dict(metrics)
        
        # After printing, emit any bounds warnings for quick visibility
        try:
            self._warn_on_out_of_bounds(metrics)
        except Exception:
            # Never break training on bounds warning issues
            pass

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

    def _collect_metrics(self, trainer: pl.Trainer) -> Dict[str, Any]:
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

    def _get_wandb_run_url(self, trainer: pl.Trainer) -> str | None:
        """Return the W&B run URL from the trainer's logger.

        Falls back to constructing the URL from run attributes if needed.
        """
        def _from_exp(exp) -> str | None:
            if exp is None:
                return None
            # Preferred: SDK-provided URL
            url = getattr(exp, "url", None)
            if url:
                return str(url)
            # Fallback: build from known attributes
            try:
                run_id = getattr(exp, "id", None) or getattr(exp, "name", None)
                entity = getattr(exp, "entity", None)
                project = getattr(exp, "project", None)
                if run_id and entity and project:
                    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
            except Exception:
                pass
            return None

        try:
            logger = getattr(trainer, "logger", None)
            if logger is None:
                return None

            # Single logger case (WandbLogger)
            url = _from_exp(getattr(logger, "experiment", None))
            if url:
                return url

            # Logger collection case
            loggers = getattr(logger, "loggers", None)
            if isinstance(loggers, (list, tuple)):
                for lg in loggers:
                    url = _from_exp(getattr(lg, "experiment", None))
                    if url:
                        return url
        except Exception:
            return None
        return None

    def _format_wandb_url_line(self, url: str) -> str:
        """Return just the plain W&B run URL (no prefixes/parentheses)."""
        return url

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

    def _warn_on_out_of_bounds(self, current_metrics: Dict[str, Any]) -> None:
        """Emit warnings for metrics outside configured min/max bounds.

        Uses bounds loaded from metrics.yaml via utils.metrics.get_metric_bounds().
        Checks namespaced keys (e.g., 'train/approx_kl').
        """
        bounds = getattr(self, "_metric_bounds", {}) or {}
        if not bounds:
            return
        for key, val in current_metrics.items():
            try:
                if key not in bounds:
                    continue
                v = self._to_python_scalar(val)
                if not self._is_number(v):
                    continue
                b = bounds.get(key, {})
                has_min = "min" in b
                has_max = "max" in b
                below = has_min and (float(v) < float(b["min"]))
                above = has_max and (float(v) > float(b["max"]))
                if not (below or above):
                    continue
                rng = [str(b.get("min")) if has_min else "-inf", str(b.get("max")) if has_max else "+inf"]
                print(f"⚠️  BOUNDS WARNING: {key}={v} outside [{rng[0]}, {rng[1]}]")
            except Exception:
                # Continue on any issues evaluating a single metric's bounds
                continue
