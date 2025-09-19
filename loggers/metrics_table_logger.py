from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pytorch_lightning.loggers.logger import (
    Logger as LightningLoggerBase,  # type: ignore
)

from utils.dict_utils import (
    group_by_namespace as group_dict_by_key_namespace,
)
from utils.dict_utils import (
    order_namespaces as order_grouped_namespaces,
)
from utils.dict_utils import (
    sort_subkeys_by_priority as sort_grouped_subkeys_by_priority,
)
from utils.formatting import is_number, number_to_string
from utils.logging import ansi as _ansi
from utils.logging import apply_ansi_background as _apply_bg
from utils.logging import strip_ansi_codes as _strip_ansi
from utils.metrics_config import metrics_config
from utils.metrics_monitor import MetricsMonitor
from utils.reports import sparkline as _sparkline
from utils.torch import to_python_scalar as _to_python_scalar


# TODO: REFACTOR this file
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

class MetricsTableLogger(LightningLoggerBase):
    """
    Lightning logger that pretty-prints the latest logged metrics as a
    namespaced table, leveraging the same formatting/highlighting rules as
    the previous PrintMetricsCallback but without re-collecting from the
    trainer — it consumes exactly what Lightning dispatches to loggers.

    Intended to be used alongside other loggers (e.g., WandbLogger,
    CsvLightningLogger) so all receive the same metrics payload.
    """
    
    # TODO: this method is a bit clunky, organize it bette, group related properties together and comment them as a group, remove any unused properties
    def __init__(
        self,
        *,
        metrics_monitor: MetricsMonitor,
        min_table_width: int = 90,
        min_value_column_width: int = 15,
        chart_column_width: int | None = None,
    ) -> None:
        """Create a pretty-print logger with sensible defaults.

        When explicit rule/formatting dicts are not provided, defaults are
        sourced from `utils.metrics_config.metrics_config` so callers can simply
        instantiate `PrintMetricsLogger()` without wiring config plumbing.
        """
        from utils.metrics_config import metrics_config

        # -- Core metadata --
        self._name = "print"
        self._version: str | int = "0"

        # -- Dependencies --
        self.metrics_monitor = metrics_monitor

        # -- Output environment --
        self.stream = sys.stdout
        self.colors_enabled: bool = sys.stdout.isatty()
        self.use_ansi_inplace: bool = False
        self.indent: int = 4

        # -- Table layout --
        self.min_table_width = int(min_table_width)
        self.min_val_width = int(min_value_column_width)
        self.sparkline_width: int = 32
        self.sparkline_history_cap: int = 512
        self.chart_column_width = self.sparkline_width

        # -- Ordering/prioritization --
        # Accept unnamespaced priorities from config and expand to full keys
        # for common namespaces so per-section sorting is preserved.
        self.key_priority: Tuple[str, ...] = tuple(metrics_config.key_priority())
        _namespaces = ("train", "val", "test")
        self._key_priority_map: Dict[str, int] = {
            f"{ns}/{sub}": idx
            for idx, sub in enumerate(self.key_priority)
            for ns in _namespaces
        }
        self.group_keys_order: Tuple[str, ...] | None = ("train", "val")
        self.group_subkeys_order: bool = True

        # -- Formatting and thresholds --
        self.delta_tol: float = 1e-12
        # Highlight behavior is resolved on-the-fly via helper methods that
        # consult the metrics config lazily (avoids upfront extraction/copy).
        self._highlight_value_bold_cache: Optional[frozenset[str]] = None
        self._highlight_row_metrics_cache: Optional[frozenset[str]] = None
        self.bgcolor_highlight: str = "bg_blue"
        self.bgcolor_alert: str = "bg_yellow"
        self.better_when_increasing: Dict[str, bool] = {}

        # -- State (updated as we log) --
        self.previous_metrics: Dict[str, Any] = {}
        self._prev: Optional[Dict[str, Any]] = None
        self._last_height: int = 0
        self._history: Dict[str, List[float]] = {}


    # --- Lightning Logger API ---
    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return self._name

    @property
    def version(self) -> str | int:  # pragma: no cover - trivial
        return self._version

    def log_hyperparams(self, params: Any) -> None:  # pragma: no cover - unused
        return None

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        assert metrics, "metrics cannot be empty"
        metrics_config.assert_metrics_within_bounds(metrics) # TODO: move this to the logger?

        # Convert values to basic Python scalars for rendering/validation 
        # and discard non-namespace keys (eg: epoch injected by Lightning)
        simple: Dict[str, Any] = {k: _to_python_scalar(v) for k, v in dict(metrics).items() if "/" in k}

        # Sticky display: merge with previous known metrics so missing keys
        # keep their last values when printing.
        merged: Dict[str, Any] = dict(self.previous_metrics)
        merged.update(simple)

        # Render the metrics table
        self._render_table(merged)

        # Track across calls
        self.previous_metrics = dict(merged)

    # TODO: remove?
    def finalize(self, status: str) -> None:  # pragma: no cover - best-effort noop
        return None

    # TODO: remove?
    def after_save_checkpoint(self, checkpoint_callback: Any) -> None:  # pragma: no cover - unused
        return None

    def _calc_deltas(self, metrics: Dict[str, Any]) -> Dict[str, Tuple[str, Optional[str]]]:
        """Compute deltas and validate delta rules in a single pass.

        Returns a mapping of full metric key -> (delta_str, color_name or None).
        Missing/invalid deltas map to ("", None).
        """
        deltas: Dict[str, Tuple[str, Optional[str]]] = {}

        # Choose previous snapshot to compare against
        prev_snapshot = self._prev if self._prev is not None else self.previous_metrics

        # Pre-validate against configured delta rules (rules are defined per bare metric name)
        for full_key, current_value in metrics.items():
            # Ensure keys are properly namespaced and valid
            assert metrics_config.is_namespaced_metric(full_key), f"Invalid metric key '{full_key}'"

            # Lookup previous value for the same full key
            if full_key not in prev_snapshot:
                # No previous value → no delta
                deltas[full_key] = ("", None)
                continue

            # If either is non-numeric, we don't compute a delta
            prev_val = prev_snapshot[full_key]
            assert is_number(current_value) and is_number(prev_val), f"Invalid metric value '{current_value}' or '{prev_val}'"

            # Apply delta rule if one exists for the bare metric name
            rule_fn = metrics_config.delta_rules_for_metric(full_key)
            if rule_fn is not None:
                assert rule_fn(prev_val, current_value), (
                    f"Delta rule violation for '{full_key}': previous={prev_val}, current={current_value}."
                )

            # Compute display delta
            delta = float(current_value) - float(prev_val)
            if abs(delta) <= self.delta_tol:
                deltas[full_key] = ("→0", "gray")
                continue

            arrow = "↑" if delta > 0 else "↓"
            inc_better = self.better_when_increasing.get(full_key, True)
            improved = (delta > 0) if inc_better else (delta < 0)
            color = "green" if improved else "red"
            mag = number_to_string(
                abs(delta),
                precision=metrics_config.precision_for_metric(full_key),
                humanize=True,
            )
            deltas[full_key] = (f"{arrow}{mag}", color)

        # For any metric without previous, ensure there's a default entry
        for k in metrics.keys():
            if k not in deltas:
                deltas[k] = ("", None)

        return deltas

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

    def _sort_key(self, namespace: str, subkey: str) -> Tuple[int, object]:
        """Sorting helper that honours an explicit key priority list."""
        full_key = metrics_config.add_namespace_to_metric(namespace, subkey)
        priority_index = self._key_priority_map.get(full_key)
        if priority_index is not None:
            return (0, priority_index)
        return (1, subkey.lower())

    def _prepare_sections(
        self,
        grouped_metrics: Dict[str, Dict[str, Any]],
        group_key_order: List[str],
        active_alerts: Dict[str, Iterable[object]],
        deltas_map: Dict[str, Tuple[str, Optional[str]]],
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
                full_key = metrics_config.add_namespace_to_metric(group_key, metric_name)
                metric_precision = metrics_config.precision_for_metric(metric_name)

                metric_value_s = number_to_string(metric_value, precision=metric_precision, humanize=True)
                if metrics_config.style_for_metric(metric_name).get("bold"):
                    metric_value_s = _ansi(metric_value_s, "bold", enable=self.colors_enabled)

                delta_str, color_name = deltas_map.get(full_key, ("", None))
                if delta_str:
                    delta_disp = _ansi(delta_str, color_name, enable=self.colors_enabled)
                    val_disp = f"{metric_value_s} {delta_disp}"
                else:
                    val_disp = metric_value_s

                chart_disp = self._spark_for_key(full_key, self.sparkline_width)

                _alerts = active_alerts.get(full_key, [])
                def _alert_message(a: object) -> str:
                    # Supports MetricAlert dataclass and plain strings (tests)
                    try:
                        from utils.metrics_monitor import MetricAlert as _MetricAlert
                        if isinstance(a, _MetricAlert):
                            return a.message
                    except Exception:
                        pass
                    if isinstance(a, str):
                        return a
                    return str(a)
                alerts_str = " | ".join([_alert_message(alert) for alert in _alerts if _alert_message(alert)])
                alert_disp = _ansi(f"⚠️  {alerts_str}", "yellow", enable=self.colors_enabled) if alerts_str else ""

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

    def _compose_lines(
        self,
        group_key_order: List[str],
        prepared: _PreparedSections,
        dimensions: _TableDimensions,
        active_alerts: Dict[str, Iterable[object]],
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
            lines.append(_ansi(header_line, "bold", enable=self.colors_enabled)) # TODO: what is self.color

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
                namespaced_metric_name = metrics_config.add_namespace_to_metric(group_key, metric_name)
                alert_active = namespaced_metric_name in active_alerts
                key_cell = f"{metric_name:<{key_width}}"
                
                highlight = False
                if alert_active:
                    highlight = True
                    row_bg_color = self.bgcolor_alert

                if not highlight and metrics_config.style_for_metric(metric_name).get("highlight"):
                    key_cell = _ansi(key_cell, "bold", enable=self.colors_enabled)
                    highlight = True
                    row_bg_color = self.bgcolor_highlight
                    
                key_padding_s = " " * self.indent
                row = (
                    f"| {key_padding_s}{key_cell} | {metric_value_padded_s} | {chart_padded_s} | {alert_padded} |"
                )

                if highlight:
                    enable_bg = self.colors_enabled or alert_active
                    row = _apply_bg(row, row_bg_color or self.bgcolor_highlight, enable=enable_bg)
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

        # Compute deltas once and validate delta rules
        deltas_map = self._calc_deltas(metrics)

        active_alerts = self.metrics_monitor.get_active_alerts()
        prepared = self._prepare_sections(grouped_metrics, sorted_grouped_metrics, active_alerts, deltas_map)
        dims = self._compute_dimensions(prepared)
        lines = self._compose_lines(grouped_metrics, prepared, dims, active_alerts)

        self._render_lines(lines)
        self._prev = dict(metrics)
        self._last_height = len(lines)
