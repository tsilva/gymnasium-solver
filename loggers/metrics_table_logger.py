from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Prefer the real Lightning logger base when available
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
    
    def __init__(
        self,
        *,
        metrics_monitor: MetricsMonitor,
        run: Any = None,
        min_table_width: int = 90,
        min_value_column_width: int = 15,
        chart_column_width: int | None = None,
    ) -> None:
        """Create a pretty-print logger with sensible defaults."""
        # Core metadata
        self._name = "print"
        self._version: str | int = "0"

        # Dependencies
        self.metrics_monitor = metrics_monitor
        self.run = run

        # Output environment
        self.stream = sys.stdout
        self.colors_enabled: bool = sys.stdout.isatty()
        self.use_ansi_inplace: bool = False
        self.indent: int = 4

        # Table layout
        self.min_table_width = int(min_table_width)
        self.min_val_width = int(min_value_column_width)
        self.sparkline_width: int = 32
        self.sparkline_history_cap: int = 512
        self.chart_column_width = self.sparkline_width

        # Ordering/prioritization
        self.key_priority: Tuple[str, ...] = tuple(metrics_config.key_priority())
        self._key_priority_map: Dict[str, int] = {
            f"{ns}/{sub}": idx
            for idx, sub in enumerate(self.key_priority)
            for ns in ("train", "val", "test")
        }
        self.group_keys_order: Tuple[str, ...] | None = ("train", "val")
        self.group_subkeys_order: bool = True

        # Formatting and thresholds
        self.delta_tol: float = 1e-12
        self.bgcolor_highlight: str = "bg_blue"
        self.bgcolor_alert: str = "bg_yellow"
        self.better_when_increasing: Dict[str, bool] = {}

        # State (updated as we log)
        self.previous_metrics: Dict[str, Any] = {}
        self._prev: Optional[Dict[str, Any]] = None
        self._last_height: int = 0
        self._history: Dict[str, List[float]] = {}
        self._metrics_with_triggered_alerts: set[str] = set()


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
        # TODO: restore this, some are crashing due to NaN/Inf values, figure out why
        #metrics_config.assert_metrics_within_bounds(metrics) # TODO: move this to the logger?

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

    def _calc_deltas(self, metrics: Dict[str, Any]) -> Dict[str, Tuple[str, Optional[str]]]:
        """Compute deltas and validate delta rules in a single pass.

        Returns a mapping of full metric key -> (delta_str, color_name or None).
        Missing/invalid deltas map to ("", None).
        """
        prev_snapshot = self._prev if self._prev is not None else self.previous_metrics
        deltas: Dict[str, Tuple[str, Optional[str]]] = {}

        for full_key, current_value in metrics.items():
            assert metrics_config.is_namespaced_metric(full_key), f"Invalid metric key '{full_key}'"

            if full_key not in prev_snapshot:
                deltas[full_key] = ("", None)
                continue

            prev_val = prev_snapshot[full_key]
            assert is_number(current_value) and is_number(prev_val), f"Invalid metric value '{current_value}' or '{prev_val}'"

            rule_fn = metrics_config.delta_rules_for_metric(full_key)
            if rule_fn is not None:
                assert rule_fn(prev_val, current_value), (
                    f"Delta rule violation for '{full_key}': previous={prev_val}, current={current_value}."
                )

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

        return deltas

    def _spark_for_key(self, full_key: str, width: int) -> str:
        values = self._history.get(full_key)
        if not values or len(values) < 2 or width <= 0: return ""
        return _sparkline(values, width)

    def _hyperlink(self, text: str, url: str) -> str:
        """Create a terminal hyperlink using OSC 8 ANSI escape sequence."""
        if not self.colors_enabled:
            return text
        return f"\x1b]8;;{url}\x1b\\{text}\x1b]8;;\x1b\\"

    def _pad_right(self, text: str, width: int) -> str:
        """Right-align text within given width (accounts for ANSI codes)."""
        clean = _strip_ansi(text)
        padding = " " * (width - len(clean))
        return padding + text

    def _pad_left(self, text: str, width: int) -> str:
        """Left-align text within given width (accounts for ANSI codes)."""
        clean = text[:width]
        return f"{clean:<{width}}"

    def _extract_alert_message(self, alert: object) -> str:
        """Extract alert message from MetricAlert or string."""
        try:
            from utils.metrics_monitor import MetricAlert as _MetricAlert
            if isinstance(alert, _MetricAlert):
                return alert._id
        except Exception:
            pass
        return str(alert) if not isinstance(alert, str) else alert

    def _format_alert_display(self, alerts: Iterable[object]) -> str:
        """Format alerts as a colored warning string."""
        messages = [self._extract_alert_message(a) for a in alerts]
        alerts_str = " | ".join([m for m in messages if m])
        return _ansi(f"⚠️  {alerts_str}", "yellow", enable=self.colors_enabled) if alerts_str else ""

    def _format_row_cells(
        self,
        metric_name: str,
        metric_value_s: str,
        chart_s: str,
        alerts_s: str,
        dimensions: _TableDimensions,
    ) -> Tuple[str, str, str]:
        """Format and pad the three main cells of a metrics row."""
        metric_padded = self._pad_right(metric_value_s, dimensions.val_width)
        chart_padded = self._pad_left(chart_s, self.chart_column_width)
        alert_padded = self._pad_right(alerts_s, dimensions.alert_width)
        return metric_padded, chart_padded, alert_padded

    def _apply_row_highlight(
        self,
        row: str,
        group_key: str,
        metric_name: str,
        alert_active: bool,
    ) -> str:
        """Apply background highlighting to a row based on alerts or config."""
        highlight = False
        row_bg_color = None

        if alert_active:
            highlight = True
            row_bg_color = self.bgcolor_alert
        elif metrics_config.style_for_metric(metric_name).get("highlight"):
            highlight = True
            row_bg_color = self.bgcolor_highlight

        if highlight:
            enable_bg = self.colors_enabled or alert_active
            row = _apply_bg(row, row_bg_color or self.bgcolor_highlight, enable=enable_bg)
            if alert_active and not enable_bg:
                row = f"⚠️  {row}"

        return row

    def _format_header_line(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Format a header line showing run ID, FPS, and time elapsed."""
        if not self.run:
            return None

        # Format run ID with hyperlink to wandb if available
        run_id = self.run.id
        try:
            import wandb
            if wandb.run is not None and wandb.run.url:
                run_id = self._hyperlink(run_id, wandb.run.url)
        except Exception:
            pass

        parts = [f"Run: {run_id}"]

        if (roll_fps := metrics.get("train/roll/fps")) is not None:
            parts.append(f"roll/fps: {number_to_string(roll_fps, precision=0, humanize=True)}")

        if (sys_fps := metrics.get("train/sys/timing/fps")) is not None:
            parts.append(f"sys/fps: {number_to_string(sys_fps, precision=0, humanize=True)}")

        if (time_elapsed := metrics.get("train/sys/timing/time_elapsed")) is not None:
            elapsed = int(time_elapsed)
            parts.append(f"elapsed: {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")

        return " | ".join(parts)

    def _update_history(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if not is_number(v): continue
            hist = self._history.setdefault(k, [])
            hist.append(float(v))
            if len(hist) > self.sparkline_history_cap:
                self._history[k] = hist[-self.sparkline_history_cap:]

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
                val_disp = f"{metric_value_s} {_ansi(delta_str, color_name, enable=self.colors_enabled)}" if delta_str else metric_value_s

                chart_disp = self._spark_for_key(full_key, self.sparkline_width)
                alert_disp = self._format_alert_display(active_alerts.get(full_key, []))

                formatted_sub[metric_name] = val_disp
                charts_sub[metric_name] = chart_disp
                alerts_sub[metric_name] = alert_disp

                val_candidates.append(_strip_ansi(val_disp))
                alert_candidates.append(_strip_ansi(alert_disp))

            formatted[group_key] = formatted_sub
            charts[group_key] = charts_sub
            # Ensure alerts collected for this group are preserved so they can
            # be rendered later in _compose_lines. Without this, the alert
            # column remains empty even when active alerts exist.
            alerts[group_key] = alerts_sub

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
        val_width = max(max((len(v) for v in prepared.val_candidates), default=0), self.min_val_width)
        alert_width = max((len(v) for v in prepared.alert_candidates), default=0)

        def _sample_row(width: int) -> str:
            return f"| {' ' * (self.indent + key_width)} | {' ' * width} | {' ' * self.chart_column_width} | {' ' * alert_width} |"

        sample = _sample_row(val_width)
        if len(sample) < self.min_table_width:
            val_width += self.min_table_width - len(sample)
            sample = _sample_row(val_width)

        return _TableDimensions(
            key_width=key_width,
            val_width=val_width,
            alert_width=alert_width,
            border="-" * len(sample),
        )

    def _compose_lines(
        self,
        group_key_order: List[str],
        prepared: _PreparedSections,
        dimensions: _TableDimensions,
        active_alerts: Dict[str, Iterable[object]],
        header_line: Optional[str] = None,
    ) -> List[str]:
        lines: List[str] = [""]

        if header_line:
            lines.extend([header_line, ""])

        key_width = dimensions.key_width
        value_width = dimensions.val_width
        alert_width = dimensions.alert_width

        for group_key in group_key_order:
            metrics_formatted_map = prepared.formatted.get(group_key)
            if not metrics_formatted_map:
                continue

            metrics_chart_map = prepared.charts.get(group_key, {})
            metrics_alerts_map = prepared.alerts.get(group_key, {})

            alert_header = "alert" if alert_width else ""
            section_header = (
                f"| {group_key + '/':<{self.indent + key_width}} | "
                f"{'':>{value_width}} | "
                f"{'':<{self.chart_column_width}} | "
                f"{alert_header:<{alert_width}} |"
            )
            lines.append(_ansi(section_header, "bold", enable=self.colors_enabled))

            for metric_name, metric_value_s in metrics_formatted_map.items():
                chart_s = metrics_chart_map.get(metric_name, "")
                alerts_s = metrics_alerts_map.get(metric_name, "")

                metric_padded, chart_padded, alert_padded = self._format_row_cells(
                    metric_name, metric_value_s, chart_s, alerts_s, dimensions
                )

                namespaced_metric_name = metrics_config.add_namespace_to_metric(group_key, metric_name)
                alert_active = namespaced_metric_name in active_alerts
                key_cell = f"{metric_name:<{key_width}}"
                if not alert_active and metrics_config.style_for_metric(metric_name).get("highlight"):
                    key_cell = _ansi(key_cell, "bold", enable=self.colors_enabled)

                key_padding = " " * self.indent
                row = f"| {key_padding}{key_cell} | {metric_padded} | {chart_padded} | {alert_padded} |"
                row = self._apply_row_highlight(row, group_key, metric_name, alert_active)
                lines.append(row)

        lines.append(dimensions.border)
        return lines

    def _render_table(self, metrics: Dict[str, Any]) -> None:
        assert metrics, "metrics cannot be empty"

        # Add metrics to the logger history (eg: used for sparklines)
        self._update_history(metrics)

        # Get active alerts and track any newly triggered alerts for this session
        active_alerts = self.metrics_monitor.get_active_alerts()
        self._metrics_with_triggered_alerts.update(active_alerts.keys())

        # Format header line before filtering (needs access to header metrics)
        header_line = self._format_header_line(metrics)

        # Define header metrics that should not appear in the table body
        header_metrics = {
            "train/roll/fps",
            "train/sys/timing/fps",
            "train/sys/timing/time_elapsed",
        }

        # Filter metrics to show those marked with show_in_table=true OR
        # any metric that has triggered an alert during this training session,
        # but exclude header metrics from the table body
        filtered_metrics = {
            k: v for k, v in metrics.items()
            if (metrics_config.show_in_table(k) or k in self._metrics_with_triggered_alerts)
            and k not in header_metrics
        }

        # If no metrics pass the filter, show all metrics as fallback (excluding header metrics)
        if not filtered_metrics:
            filtered_metrics = {k: v for k, v in metrics.items() if k not in header_metrics}

        # Group metrics by namespace (eg: train and val namespaces)
        grouped_metrics = group_dict_by_key_namespace(filtered_metrics)

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

        # active_alerts already retrieved earlier in this method
        prepared = self._prepare_sections(grouped_metrics, sorted_grouped_metrics, active_alerts, deltas_map)
        dims = self._compute_dimensions(prepared)
        lines = self._compose_lines(grouped_metrics, prepared, dims, active_alerts, header_line=header_line)

        self._render_lines(lines)
        self._prev = dict(metrics)
        self._last_height = len(lines)
