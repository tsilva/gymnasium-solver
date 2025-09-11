from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List, Iterable

import os
import sys
import numbers

from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase  # type: ignore

import torch
from utils.logging import ansi as _ansi
from utils.logging import apply_ansi_background as _apply_bg
from utils.logging import strip_ansi_codes as _strip_ansi
from utils.reports import sparkline as _sparkline
from utils.dict_utils import group_by_namespace as _group_by_namespace
from utils.torch import to_python_scalar as _to_python_scalar
from utils.formatting import (
    is_number,
    format_value as _format_value,
    format_delta_magnitude as _fmt_delta_mag,
    get_sort_key as _get_sort_key,
)

class PrintMetricsLogger(LightningLoggerBase):
    """
    Lightning logger that pretty-prints the latest logged metrics as a
    namespaced table, leveraging the same formatting/highlighting rules as
    the previous PrintMetricsCallback but without re-collecting from the
    trainer — it consumes exactly what Lightning dispatches to loggers.

    Intended to be used alongside other loggers (e.g., WandbLogger,
    CsvLightningLogger) so all receive the same metrics payload.
    """

    # ANSI codes and helpers are provided by utils.logging

    def __init__(
        self,
        *,
        metric_precision: Dict[str, int] | None = None,
        metric_delta_rules: Dict[str, Callable] | None = None,
        algorithm_metric_rules: Dict[str, dict] | None = None,
        min_val_width: int = 15,
        key_priority: List[str] | None = None,
    ) -> None:
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

        # -------- Inlined table printer configuration/state --------
        self.float_fmt: str = ".2f"
        self.indent: int = 4
        self.compact_numbers: bool = True
        self.color: bool = bool(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        self.better_when_increasing: Dict[str, bool] = {}
        self.fixed_section_order: Optional[List[str]] = ["train", "val"]
        self.sort_keys_within_section: bool = True
        self.use_ansi_inplace: bool = False
        self.stream = sys.stdout
        self.delta_tol: float = 1e-12
        self.metric_precision_map: Dict[str, int] = dict(self.metric_precision)
        # Highlight settings (bare metric subkeys)
        self.highlight_value_bold_for_set = set(_hl_value_bold_for or {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "epoch"})
        self.highlight_row_for_set = set(_hl_row_for or {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "total_timesteps"})
        self.highlight_row_bg_color: str = _hl_row_bg_color or "bg_blue"
        self.highlight_row_bold: bool = bool(_hl_row_bold)
        # Bounds-based highlighting
        self.metric_bounds_map: Dict[str, Dict[str, float]] = dict(self._metric_bounds or {})
        self.highlight_bounds_bg_color: str = 'bg_yellow'
        # In-memory history for sparklines (per full metric key)
        self._prev: Optional[Dict[str, Any]] = None
        self._last_height: int = 0
        self._history: Dict[str, List[float]] = {}
        self.show_sparklines: bool = True
        self.sparkline_width: int = 32
        self.sparkline_history_cap: int = 512

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
        # Convert values to basic Python scalars for rendering/validation
        simple: Dict[str, Any] = {k: _to_python_scalar(v) for k, v in dict(metrics).items()}

        # Sanitize current payload: drop bare keys (e.g., 'epoch') when a namespaced duplicate exists
        namespaced = set(k for k in simple.keys() if "/" in k)
        to_remove = []
        for k in list(simple.keys()):
            if "/" in k:
                continue
            # If a namespaced duplicate exists, prefer the namespaced one
            if f"train/{k}" in namespaced or f"val/{k}" in namespaced or f"test/{k}" in namespaced:
                to_remove.append(k)
        for k in to_remove:
            simple.pop(k, None)

        # Sticky display: merge with previous known metrics so missing keys
        # keep their last values when printing.
        merged: Dict[str, Any] = dict(self.previous_metrics)
        merged.update(simple)

        # Re-sanitize after merging to avoid reintroducing bare duplicates
        # when a namespaced key is present in the union.
        merged_namespaced = set(k for k in merged.keys() if "/" in k)
        merged_to_remove = []
        for k in list(merged.keys()):
            if "/" in k:
                continue
            if f"train/{k}" in merged_namespaced or f"val/{k}" in merged_namespaced or f"test/{k}" in merged_namespaced:
                merged_to_remove.append(k)
        for k in merged_to_remove:
            merged.pop(k, None)

        # Validate deltas and algorithm-specific rules using the latest snapshot
        self._validate_metric_deltas(simple)
        self._check_algorithm_metric_rules(simple)

        # Render the metrics table
        self._render_table(merged)

        # Track across calls
        self.previous_metrics = dict(merged)

    def finalize(self, status: str) -> None:  # pragma: no cover - best-effort noop
        return None

    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:  # pragma: no cover - unused
        return None

    # --- Helpers ---
    # Scalar conversion lives in utils.torch.to_python_scalar

    def _validate_metric_deltas(self, current_metrics: Dict[str, Any]) -> None:
        for metric_name, rule_lambda in self.metric_delta_rules.items():
            if metric_name in current_metrics and metric_name in self.previous_metrics:
                curr = current_metrics[metric_name]
                prev = self.previous_metrics[metric_name]
                if not (is_number(curr) and is_number(prev)):
                    continue
                ok = bool(rule_lambda(prev, curr))
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
            if not is_number(curr):
                continue
            check_func = rule_config.get('check')
            message_template = rule_config.get('message', f"Algorithm metric rule violated for '{metric_name}'")
            level = rule_config.get('level', 'warning')
            if check_func is None:
                continue
            satisfied = False
            if callable(check_func):
                if getattr(check_func, "__code__", None) and check_func.__code__.co_argcount == 1:
                    satisfied = bool(check_func(curr))
                elif metric_name in self.previous_metrics:
                    prev = self.previous_metrics.get(metric_name)
                    if is_number(prev):
                        satisfied = bool(check_func(prev, curr))
                    else:
                        satisfied = True
                else:
                    satisfied = True
            if not satisfied:
                prev = self.previous_metrics.get(metric_name, 'N/A')
                msg = message_template.format(metric_name=metric_name, current_value=curr, previous_value=prev)
                if level == 'error':
                    print(f"🚨 ALGORITHM ERROR: {msg}")
                else:
                    print(f"⚠️  ALGORITHM WARNING: {msg}")

    def _delta_for_key(self, namespace: str, key: str, value: Any):
        # In case we don't have a previous value, return empty string
        if self._prev is None: return ("", None)

        # In case we don't have a previous value for the key, return empty string
        full_key = f"{namespace}/{key}" if key else namespace
        if full_key not in self._prev: return ("", None)

        # In case the delta is less than the tolerance, return 0 delta
        prev_value = self._prev[full_key]
        assert is_number(value) and is_number(prev_value), f"Value and previous value must be numbers: {value} and {prev_value}"
        delta = float(value) - float(prev_value)
        if abs(delta) <= self.delta_tol: return ("→0", "gray")

        # Determine arrow to represent the delta direction
        arrow = "↑" if delta > 0 else "↓"

        # Determine color to represent if delta direction 
        # is better or worse (depends on metric delta rule)
        inc_better = self.better_when_increasing.get(full_key, True)
        improved = (delta > 0) if inc_better else (delta < 0)
        color = "green" if improved else "red"

        mag = _fmt_delta_mag(
            abs(delta),
            full_key,
            precision_map=self.metric_precision_map,
            compact_numbers=self.compact_numbers,
            float_fmt=self.float_fmt,
        )
        return (f"{arrow}{mag}", color)

    def _spark_for_key(self, full_key: str, width: int) -> str:
        values = self._history.get(full_key)
        if not values or len(values) < 2 or width <= 0:
            return ""
        return _sparkline(values, width)

    def _update_history(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if not is_number(v):
                continue
            val = float(v)
            hist = self._history.setdefault(k, [])
            hist.append(val)
            if len(hist) > self.sparkline_history_cap:
                self._history[k] = hist[-self.sparkline_history_cap :]

    def _render_lines(self, lines: List[str]) -> None:
        text = "\n".join(lines)
        if self.use_ansi_inplace and self._last_height > 0:
            self.stream.write(f"\x1b[{self._last_height}F")
            self.stream.write("\x1b[0J")
            self.stream.write(text + "\n")
            self.stream.flush()
        else:
            if not self.use_ansi_inplace:
                os.system('cls' if os.name == 'nt' else 'clear')
            print(text, file=self.stream)

    def _render_table(self, data: Dict[str, Any]) -> None:
        # In case we don't have any data, return
        if not data: return

        self._update_history(data)

        grouped = _group_by_namespace(data)

        ns_names = list(grouped.keys())
        if self.fixed_section_order:
            pref = [ns for ns in self.fixed_section_order if ns in grouped]
            rest = sorted([ns for ns in ns_names if ns not in self.fixed_section_order])
            ns_order = pref + rest
        else:
            ns_order = sorted(ns_names)
        for ns in ns_order:
            if self.sort_keys_within_section:
                grouped[ns] = dict(
                    sorted(grouped[ns].items(), key=lambda kv: _get_sort_key(ns, kv[0], self.key_priority))
                )
                
        formatted: Dict[str, Dict[str, str]] = {}
        val_candidates: List[str] = []
        key_candidates: List[str] = [ns + "/" for ns in ns_order]
        for ns in ns_order:
            subdict = grouped[ns]
            f_sub: Dict[str, str] = {}
            for sub, v in subdict.items():
                key_candidates.append(sub)
                full_key = f"{ns}/{sub}" if sub else ns
                val_str = _format_value(
                    v,
                    full_key,
                    precision_map=self.metric_precision_map,
                    compact_numbers=self.compact_numbers,
                    float_fmt=self.float_fmt,
                )
                if sub in self.highlight_value_bold_for_set:
                    val_str = _ansi(val_str, "bold", enable=self.color)
                delta_str, color_name = self._delta_for_key(ns, sub, v)
                if delta_str:
                    delta_disp = _ansi(delta_str, color_name, enable=self.color)
                    val_disp = f"{val_str} {delta_disp}"
                else:
                    val_disp = val_str
                if self.show_sparklines and is_number(v):
                    chart = self._spark_for_key(full_key, self.sparkline_width)
                    if chart:
                        val_disp = f"{val_disp}  {chart}"
                f_sub[sub] = val_disp
                val_candidates.append(_strip_ansi(val_disp))
            formatted[ns] = f_sub
        indent = self.indent
        key_width = max((len(k) for k in key_candidates), default=0)
        val_width = max((len(v) for v in val_candidates), default=0)
        val_width = max(val_width, self.min_val_width)
        border_len = 2 + (indent + key_width) + 3 + val_width + 2
        border = "-" * border_len
        lines: List[str] = []
        lines.append(border)
        for ns in ns_order:
            if not formatted.get(ns):
                continue
            header = ns + "/"
            header_line = f"| {header:<{indent + key_width}} | {'':>{val_width}} |"
            lines.append(_ansi(header_line, "bold", enable=self.color))
            for sub, val in formatted[ns].items():
                val_display_len = len(_strip_ansi(val))
                val_padding = val_width - val_display_len
                val_padded = (" " * val_padding + val) if val_padding > 0 else val
                key_cell = f"{sub:<{key_width}}"
                highlight = False
                row_bg_color = None
                full_key = f"{ns}/{sub}" if sub else ns
                # Priority 1: bounds-based highlight (yellow); allow bare-name lookup
                bounds = self.metric_bounds_map.get(full_key) or self.metric_bounds_map.get(sub)
                if bounds:
                    raw_val = grouped.get(ns, {}).get(sub)
                    if is_number(raw_val):
                        vnum = float(raw_val)
                        below = ("min" in bounds) and (vnum < float(bounds["min"]))
                        above = ("max" in bounds) and (vnum > float(bounds["max"]))
                        if below or above:
                            highlight = True
                            row_bg_color = self.highlight_bounds_bg_color
                # Priority 2: configured row highlight
                if not highlight and sub in self.highlight_row_for_set:
                    if self.highlight_row_bold:
                        key_cell = _ansi(key_cell, "bold", enable=self.color)
                    highlight = True
                    row_bg_color = self.highlight_row_bg_color
                row = f"| {' ' * indent}{key_cell} | {val_padded} |"
                if highlight:
                    row = _apply_bg(row, row_bg_color or self.highlight_row_bg_color, enable=self.color)
                lines.append(row)
        lines.append(border)
        self._render_lines(lines)
        self._prev = dict(data)
        self._last_height = len(lines)
