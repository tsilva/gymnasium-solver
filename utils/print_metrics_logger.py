from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase  # type: ignore

import torch


class PrintMetricsLogger(LightningLoggerBase):
    """
    Lightning logger that pretty-prints the latest logged metrics as a
    namespaced table, leveraging the same formatting/highlighting rules as
    the previous PrintMetricsCallback but without re-collecting from the
    trainer — it consumes exactly what Lightning dispatches to loggers.

    Intended to be used alongside other loggers (e.g., WandbLogger,
    CsvLightningLogger) so all receive the same metrics payload.
    """

    def __init__(
        self,
        *,
        metric_precision: Dict[str, int] | None = None,
        metric_delta_rules: Dict[str, Callable] | None = None,
        algorithm_metric_rules: Dict[str, dict] | None = None,
        min_val_width: int = 15,
        key_priority: List[str] | None = None,
    ) -> None:
        from utils.table_printer import NamespaceTablePrinter
        from utils.metrics import metrics_config

        # Display config
        self._name = "print"
        self._version: str | int = "0"
        self._experiment = None

        # Rules/config
        self.metric_precision = dict(metric_precision or {})
        self.metric_delta_rules = dict(metric_delta_rules or {})
        self.algorithm_metric_rules = dict(algorithm_metric_rules or {})
        self.min_val_width = int(min_val_width)
        self.key_priority = list(key_priority or [])
        self.previous_metrics: Dict[str, Any] = {}

        # Highlight/bounds from metrics.yaml
        _m = metrics_config
        _hl_cfg = _m.highlight_config()
        _hl_value_bold_for = _hl_cfg.get('value_bold_metrics', set())
        _hl_row_for = _hl_cfg.get('row_metrics', set())
        _hl_row_bg_color = _hl_cfg.get('row_bg_color', 'bg_blue')
        _hl_row_bold = bool(_hl_cfg.get('row_bold', True))
        _metric_bounds = _m.metric_bounds()
        self._metric_bounds = dict(_metric_bounds)

        # Dedicated table printer
        self._printer = NamespaceTablePrinter(
            compact_numbers=True,
            color=True,
            use_ansi_inplace=False,
            metric_precision=self.metric_precision,
            min_val_width=self.min_val_width,
            key_priority=self.key_priority,
            highlight_value_bold_for=_hl_value_bold_for,
            highlight_row_for=_hl_row_for,
            highlight_row_bg_color=_hl_row_bg_color,
            highlight_row_bold=_hl_row_bold,
            metric_bounds=_metric_bounds,
            highlight_bounds_bg_color='bg_yellow',
        )

    # --- Lightning Logger API ---
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    @property
    def version(self) -> str | int:  # pragma: no cover - trivial
        return self._version

    @property
    def experiment(self):  # pragma: no cover - not used
        return self._experiment

    def log_hyperparams(self, params: Any) -> None:  # pragma: no cover - unused
        return None

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        try:
            # Convert values to basic Python scalars for rendering/validation
            simple: Dict[str, Any] = {k: self._to_python_scalar(v) for k, v in dict(metrics).items()}

            # Validate deltas and algorithm-specific rules using the latest snapshot
            self._validate_metric_deltas(simple)
            self._check_algorithm_metric_rules(simple)

            # Render the metrics table
            self._printer.update(simple)

            # Track across calls
            self.previous_metrics.update(simple)
        except Exception:
            # Never block training on printing issues
            pass

    def finalize(self, status: str) -> None:  # pragma: no cover - best-effort noop
        return None

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:  # pragma: no cover - unused
        return None

    # --- Helpers ---
    def _to_python_scalar(self, x: Any) -> Any:
        try:
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.detach().item()
                return x.detach().float().mean().item()
            if hasattr(x, "item") and callable(getattr(x, "item")):
                return x.item()
            return x
        except Exception:
            return x

    def _is_number(self, x: Any) -> bool:
        import numbers
        return isinstance(x, numbers.Number)

    def _validate_metric_deltas(self, current_metrics: Dict[str, Any]) -> None:
        for metric_name, rule_lambda in self.metric_delta_rules.items():
            if metric_name in current_metrics and metric_name in self.previous_metrics:
                curr = current_metrics[metric_name]
                prev = self.previous_metrics[metric_name]
                if not (self._is_number(curr) and self._is_number(prev)):
                    continue
                try:
                    ok = bool(rule_lambda(prev, curr))
                except Exception:
                    ok = True
                if ok:
                    continue
                # Emit a clear violation message but do not raise
                print(
                    f"⚠️  Metric delta rule violation for '{metric_name}': previous={prev}, current={curr}."
                )

    def _check_algorithm_metric_rules(self, current_metrics: Dict[str, Any]) -> None:
        for metric_name, rule_config in self.algorithm_metric_rules.items():
            if metric_name not in current_metrics:
                continue
            curr = current_metrics[metric_name]
            if not self._is_number(curr):
                continue
            try:
                check_func = rule_config.get('check')
                message_template = rule_config.get('message', f"Algorithm metric rule violated for '{metric_name}'")
                level = rule_config.get('level', 'warning')
                if check_func is None:
                    continue
                satisfied = False
                if callable(check_func):
                    try:
                        if getattr(check_func, "__code__", None) and check_func.__code__.co_argcount == 1:
                            satisfied = bool(check_func(curr))
                        elif metric_name in self.previous_metrics:
                            prev = self.previous_metrics.get(metric_name)
                            if self._is_number(prev):
                                satisfied = bool(check_func(prev, curr))
                            else:
                                satisfied = True
                        else:
                            satisfied = True
                    except Exception:
                        satisfied = True
                if not satisfied:
                    prev = self.previous_metrics.get(metric_name, 'N/A')
                    msg = message_template.format(metric_name=metric_name, current_value=curr, previous_value=prev)
                    if level == 'error':
                        print(f"🚨 ALGORITHM ERROR: {msg}")
                    else:
                        print(f"⚠️  ALGORITHM WARNING: {msg}")
            except Exception:
                # Be resilient on logging
                pass

