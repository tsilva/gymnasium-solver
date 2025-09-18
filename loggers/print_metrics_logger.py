from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List, Iterable, Tuple, Mapping
from dataclasses import dataclass

import os
import sys

from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase  # type: ignore

from utils.logging import ansi as _ansi
from utils.logging import apply_ansi_background as _apply_bg
from utils.logging import strip_ansi_codes as _strip_ansi
from utils.reports import sparkline as _sparkline
from utils.dict_utils import (
    group_by_namespace as group_dict_by_key_namespace,
    order_namespaces as order_grouped_namespaces,
    sort_subkeys_by_priority as sort_grouped_subkeys_by_priority,
)
from utils.torch import to_python_scalar as _to_python_scalar
from utils.formatting import (
    is_number,
    number_to_string
)
from utils.metrics_monitor import MetricsMonitor


@dataclass
class _PreparedSections:
    formatted: Dict[str, Dict[str, str]]
    charts: Dict[str, Dict[str, str]]
    alerts: Dict[str, Dict[str, str]]
    key_candidates: List[str]
    val_candidates: List[str]
    alert_candidates: List[str]


@dataclass
class _TableDimensions:
    key_width: int
    val_width: int
    alert_width: int
    border: str

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
        metrics_monitor: MetricsMonitor,
        metric_precision: Dict[str, int] | None = None,
        metric_delta_rules: Dict[str, Callable] | None = None,
        min_val_width: int = 15,
        min_table_width: int = 90,
        chart_col_width: int | None = None,
        key_priority: List[str] | None = None,
    ) -> None:
        """Create a pretty-print logger with sensible defaults.

        When explicit rule/formatting dicts are not provided, defaults are
        sourced from `utils.metrics_config.metrics_config` so callers can simply
        instantiate `PrintMetricsLogger()` without wiring config plumbing.

        Algorithm-specific metric checks have been removed.
        """
        from utils.metrics_config import metrics_config

        # Display config
        self._name = "print"
        self._version: str | int = "0"
        self._experiment = None

        # Rules/config (populate from metrics_config when not provided)
        default_precision = metrics_config.metric_precision_dict()
        default_delta_rules = metrics_config.metric_delta_rules()
        default_key_priority = metrics_config.key_priority() or []

        self.metrics_monitor = metrics_monitor

        self.metric_precision = dict(metric_precision or default_precision)
        self.metric_delta_rules = dict(metric_delta_rules or default_delta_rules)
        self.min_val_width = int(min_val_width)
        # Enforce a minimum overall table width to reduce jitter from
        # fluctuating value lengths (numbers + deltas + sparklines).
        self.min_table_width = int(min_table_width)
        # Fixed-width charts column; default matches sparkline width
        self.chart_column_width = int(chart_col_width or 0) if chart_col_width is not None else 0
        self.key_priority = list(key_priority or list(default_key_priority))
        self._key_priority_map: Dict[str, int] = {key: idx for idx, key in enumerate(self.key_priority)}
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
        self.indent: int = 4
        self.compact_numbers: bool = True
        self.color: bool = bool(sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        self.better_when_increasing: Dict[str, bool] = {}
        self.group_keys_order: Optional[List[str]] = ["train", "val"]
        self.group_subkeys_order: bool = True
        self.use_ansi_inplace: bool = False
        self.stream = sys.stdout
        self.delta_tol: float = 1e-12
        self.metric_precision_map: Dict[str, int] = dict(self.metric_precision)
        # Highlight settings (bare metric subkeys)
        self.style_bold_metrics_set = set(_hl_value_bold_for or {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "epoch"})
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
       
        # If chart_col_width not explicitly set, mirror sparkline_width
        if self.chart_column_width == 0:
            self.chart_column_width = int(self.sparkline_width)

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
        assert metrics, "Metrics cannot be empty"
        
        # Convert values to basic Python scalars for rendering/validation 
        # and discard non-namespace keys (eg: epoch injected by Lightning)
        simple: Dict[str, Any] = {k: _to_python_scalar(v) for k, v in dict(metrics).items() if "/" in k}

        # Sticky display: merge with previous known metrics so missing keys
        # keep their last values when printing.
        merged: Dict[str, Any] = dict(self.previous_metrics)
        merged.update(simple)

        # Validate deltas using the latest snapshot
        self._validate_metric_deltas(simple)

        # TODO: assert metrics within bounds

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

                # Assert that the value and previous value are numbers
                assert is_number(curr) and is_number(prev), f"Value and previous value must be numbers: {curr} and {prev}"

                # If the delta rule is satisfied, continue
                ok = bool(rule_lambda(prev, curr))
                if ok: continue

                # Emit a clear violation message but do not raise
                print(
                    f"⚠️  Metric delta rule violation for '{metric_name}': previous={prev}, current={curr}."
                )

    # Algorithm-specific metric rules removed.

    def _delta_for_key(self, namespace: str, key: str, value: Any):
        # In case we don't have a previous value, return empty string
        if self._prev is None: return ("", None)

        # In case we don't have a previous value for the key, return empty string
        full_key = self._full_key(namespace, key)
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

        # Format the delta magnitude
        mag = number_to_string(
            abs(delta),
            precision=self.metric_precision_map.get(full_key, 2),
            humanize=True,
        )
        return (f"{arrow}{mag}", color)

    def _spark_for_key(self, full_key: str, width: int) -> str:
        values = self._history.get(full_key)
        if not values or len(values) < 2 or width <= 0: return ""
        return _sparkline(values, width)

    def _update_history(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            # Skip if the value is not a number
            if not is_number(v): continue

            # Add the value to the history
            val = float(v)
            hist = self._history.setdefault(k, [])
            hist.append(val)

            # Skip if the history is less than the capacity
            if len(hist) <= self.sparkline_history_cap: continue

            # Set the history for the key
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

    @staticmethod
    def _full_key(namespace: str, subkey: Optional[str]) -> str:
        return f"{namespace}/{subkey}" if subkey else namespace

    @staticmethod
    def _subkey_from_full(full_key: str) -> str:
        return full_key.rsplit("/", 1)[-1]

    def _bounds_for_metric(self, full_key: str) -> Optional[Mapping[str, Any]]:
        bounds = self.metric_bounds_map.get(full_key)
        if bounds:
            return bounds
        subkey = self._subkey_from_full(full_key)
        return self.metric_bounds_map.get(subkey)

    def _sort_key(self, namespace: str, subkey: str) -> Tuple[int, object]:
        """Sorting helper that honours an explicit key priority list."""
        full_key = self._full_key(namespace, subkey)
        priority_index = self._key_priority_map.get(full_key)
        if priority_index is not None:
            return (0, priority_index)
        return (1, subkey.lower())

    def _sort_grouped_metrics(self, grouped_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        group_keys = list(grouped_metrics.keys())
        if not self.group_keys_order: return sorted(group_keys)
        pref = [key for key in self.group_keys_order if key in group_keys]
        rest = sorted(key for key in group_keys if key not in self.group_keys_order)
        return pref + rest

    def _prepare_sections(
        self,
        grouped_metrics: Dict[str, Dict[str, Any]],
        group_key_order: List[str],
        active_alerts: Dict[str, Iterable[Dict[str, str]]],
    ) -> _PreparedSections:
        formatted: Dict[str, Dict[str, str]] = {}
        charts: Dict[str, Dict[str, str]] = {}
        key_candidates: List[str] = [f"{ns}/" for ns in group_key_order]
        val_candidates: List[str] = []
        alert_candidates: List[str] = []
        alerts: Dict[str, Dict[str, str]] = {}

        for group_key in group_key_order:
            group_metrics = grouped_metrics.get(group_key, {})
            assert group_metrics, f"Group metrics cannot be empty: {group_key}"

            formatted_sub: Dict[str, str] = {}
            charts_sub: Dict[str, str] = {}
            alerts_sub: Dict[str, str] = {}

            for metric_name, metric_value in group_metrics.items():
                key_candidates.append(metric_name)
                full_key = self._full_key(group_key, metric_name)
                metric_precision = self.metric_precision_map.get(metric_name, 2)

                metric_value_s = number_to_string(metric_value, precision=metric_precision, humanize=True)
                if metric_name in self.style_bold_metrics_set:
                    metric_value_s = _ansi(metric_value_s, "bold", enable=self.color)

                delta_str, color_name = self._delta_for_key(group_key, metric_name, metric_value)
                if delta_str:
                    delta_disp = _ansi(delta_str, color_name, enable=self.color)
                    val_disp = f"{metric_value_s} {delta_disp}"
                else:
                    val_disp = metric_value_s

                chart_disp = self._spark_for_key(full_key, self.sparkline_width)

                _alerts = active_alerts.get(full_key, [])
                alerts_str = " | ".join([alert['message'] for alert in _alerts])
                alert_disp = _ansi(f"⚠️  {alerts_str}", "yellow", enable=self.color) if alerts_str else ""

                formatted_sub[metric_name] = val_disp
                charts_sub[metric_name] = chart_disp
                alerts_sub[metric_name] = alert_disp

                val_candidates.append(_strip_ansi(val_disp))
                alert_candidates.append(_strip_ansi(alert_disp))

            formatted[group_key] = formatted_sub
            charts[group_key] = charts_sub

        return _PreparedSections(
            formatted=formatted,
            charts=charts,
            alerts=alerts,
            key_candidates=key_candidates,
            val_candidates=val_candidates,
            alert_candidates=alert_candidates,
        )

    def _compute_dimensions(self, prepared: _PreparedSections) -> _TableDimensions:
        key_width = max((len(k) for k in prepared.key_candidates), default=0)
        val_width = max((len(v) for v in prepared.val_candidates), default=0)
        val_width = max(val_width, self.min_val_width)
        alert_width = max((len(v) for v in prepared.alert_candidates), default=0)

        def _sample_row(width: int) -> str:
            return (
                f"| {' ' * (self.indent + key_width)} | "
                f"{' ' * width} | "
                f"{' ' * self.chart_column_width} | "
                f"{' ' * alert_width} |"
            )

        sample = _sample_row(val_width)
        if len(sample) < self.min_table_width:
            val_width += self.min_table_width - len(sample)
            sample = _sample_row(val_width)

        border = "-" * len(sample)
        return _TableDimensions(
            key_width=key_width,
            val_width=val_width,
            alert_width=alert_width,
            border=border,
        )

    def _resolve_row_highlight(
        self,   
        subkey: str,
        key_cell: str,
        alert_active: bool,
    ) -> Tuple[str, bool, Optional[str]]:
        highlight = False
        row_bg_color: Optional[str] = None

        if alert_active:
            highlight = True
            row_bg_color = self.highlight_bounds_bg_color

        if not highlight and subkey in self.highlight_row_for_set:
            if self.highlight_row_bold:
                key_cell = _ansi(key_cell, "bold", enable=self.color)
            highlight = True
            row_bg_color = self.highlight_row_bg_color

        return key_cell, highlight, row_bg_color

    def _compose_lines(
        self,
        grouped_metrics: Dict[str, Dict[str, Any]],
        group_key_order: List[str],
        prepared: _PreparedSections,
        dimensions: _TableDimensions,
        active_alerts: Dict[str, Iterable[str]],
    ) -> List[str]:
        lines: List[str] = [""]
        key_width = dimensions.key_width
        value_width = dimensions.val_width
        alert_width = dimensions.alert_width

        for group_key in group_key_order:
            metrics_formatted_map = prepared.formatted.get(group_key)
            if not metrics_formatted_map:
                continue

            metrics_chart_map = prepared.charts.get(group_key, {})
            metrics_alerts_map = prepared.alerts.get(group_key, {})

            header = f"{group_key}/"
            alert_header = "alert" if alert_width else ""
            header_line = (
                f"| {header:<{self.indent + key_width}} | "
                f"{'':>{value_width}} | "
                f"{'':<{self.chart_column_width}} | "
                f"{alert_header:<{alert_width}} |"
            )
            lines.append(_ansi(header_line, "bold", enable=self.color)) # TODO: what is self.color

            for metric_name, metric_value_s in metrics_formatted_map.items():
                # Pad metric value to the right
                metric_value_clean_s = _strip_ansi(metric_value_s)
                metric_value_len = len(metric_value_clean_s)
                metric_value_padding_s = " " * (value_width - metric_value_len)
                metric_value_padded_s = metric_value_padding_s + metric_value_s

                # Pad chart to the left
                chart_s = metrics_chart_map.get(metric_name, "")
                chart_clean_s = chart_s[: self.chart_column_width]
                chart_padded_s = f"{chart_clean_s:<{self.chart_column_width}}"

                # Pad alert to the right
                alerts_s = metrics_alerts_map.get(metric_name, "")
                alert_clean = _strip_ansi(alerts_s)
                alert_len = len(alert_clean)
                alert_padding_s = " " * (alert_width - alert_len)
                alert_padded = alert_padding_s + alerts_s

                # Resolve row highlight
                full_key = self._full_key(group_key, metric_name)
                alert_active = full_key in active_alerts
                key_cell = f"{metric_name:<{key_width}}"
                
                key_cell, highlight, row_bg_color = self._resolve_row_highlight(metric_name, key_cell, alert_active)    

                key_padding_s = " " * self.indent
                row = (
                    f"| {key_padding_s}{key_cell} | {metric_value_padded_s} | {chart_padded_s} | {alert_padded} |"
                )

                if highlight:
                    enable_bg = self.color or alert_active
                    row = _apply_bg(row, row_bg_color or self.highlight_row_bg_color, enable=enable_bg)
                    if alert_active and not enable_bg:
                        row = f"⚠️  {row}"

                # Add row to lines
                lines.append(row)

        # Add border to lines
        lines.append(dimensions.border)

        # Return lines
        return lines

    def _render_table(self, metrics: Dict[str, Any]) -> None:
        assert metrics, "metrics cannot be empty"

        # Add metrics to the logger history (eg: used for sparklines)
        self._update_history(metrics)

        # Group metrics by namespace (eg: train and val namespaces)
        grouped_metrics = group_dict_by_key_namespace(metrics)

        # Order namespaces using reusable util (prefers self.group_keys_order)
        sorted_grouped_metrics = order_grouped_namespaces(grouped_metrics, self.group_keys_order)

        # Sort subkeys per-namespace using reusable util (respects priority map)
        if self.group_subkeys_order:
            grouped_metrics = sort_grouped_subkeys_by_priority(
                grouped_metrics,
                sorted_grouped_metrics,
                self._key_priority_map,
            )

        active_alerts = self.metrics_monitor.get_active_alerts()
        prepared = self._prepare_sections(grouped_metrics, sorted_grouped_metrics, active_alerts)
        dims = self._compute_dimensions(prepared)
        lines = self._compose_lines(grouped_metrics, sorted_grouped_metrics, prepared, dims, active_alerts)

        self._render_lines(lines)
        self._prev = dict(metrics)
        self._last_height = len(lines)
