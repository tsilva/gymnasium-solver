from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List, Iterable, Tuple

import os
import sys

from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase  # type: ignore

from utils.logging import ansi as _ansi
from utils.logging import apply_ansi_background as _apply_bg
from utils.logging import strip_ansi_codes as _strip_ansi
from utils.reports import sparkline as _sparkline
from utils.dict_utils import group_by_namespace as _group_by_namespace
from utils.torch import to_python_scalar as _to_python_scalar
from utils.formatting import (
    is_number,
    number_to_string
)
from utils.metrics_monitor import MetricsMonitor

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
        self.chart_col_width = int(chart_col_width or 0) if chart_col_width is not None else 0
        self.key_priority = list(key_priority or list(default_key_priority))
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
        # Persisted bounds-alert messages by full metric key. Alerts remain
        # until we observe a non-violating value reported for that key.
        self._active_bounds_alerts: Dict[str, str] = {}
       
        # If chart_col_width not explicitly set, mirror sparkline_width
        if self.chart_col_width == 0:
            self.chart_col_width = int(self.sparkline_width)

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
        # and discard non-namespace keys (eg: epoch injected by Lightning)
        simple: Dict[str, Any] = {k: _to_python_scalar(v) for k, v in dict(metrics).items() if "/" in k}

        # Sticky display: merge with previous known metrics so missing keys
        # keep their last values when printing.
        merged: Dict[str, Any] = dict(self.previous_metrics)
        merged.update(simple)

        # Validate deltas using the latest snapshot
        self._validate_metric_deltas(simple)

        # Update persisted bounds alerts based only on keys reported now
        # (absence of a key does not clear its alert).
        self._update_bounds_alerts(simple)

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

    def _update_bounds_alerts(self, current_snapshot: Dict[str, Any]) -> None:
        """Update active bounds alerts based on the current metrics payload.

        - Adds/updates alerts for keys present in `current_snapshot` that
          violate configured bounds.
        - Clears alerts only when a key is present and within bounds.
        - Leaves alerts untouched if a key is absent (persist across epochs).
        """
        for full_key, val in current_snapshot.items():
            if not is_number(val):
                continue
            # Lookup bounds for fully-qualified key, then bare subkey
            bounds = self.metric_bounds_map.get(full_key)
            if not bounds:
                subkey = full_key.rsplit("/", 1)[-1]
                bounds = self.metric_bounds_map.get(subkey)
            if not bounds:
                # No bounds configured; do not change any alert state
                continue

            v = float(val)
            has_min = "min" in bounds
            has_max = "max" in bounds
            below = has_min and (v < float(bounds["min"]))
            above = has_max and (v > float(bounds["max"]))
            violating = below or above

            if violating:
                # Format a concise message and persist it
                rng = None
                if has_min and has_max:
                    rng = f"[{bounds['min']}, {bounds['max']}]"
                elif has_min:
                    rng = f">= {bounds['min']}"
                elif has_max:
                    rng = f"<= {bounds['max']}"
                else:
                    rng = "(no bounds)"

                prec = self.metric_precision_map.get(full_key, self.metric_precision_map.get(full_key.rsplit("/", 1)[-1], 2))
                vdisp = number_to_string(v, precision=prec, humanize=True)
                msg = f"⚠️  Bounds alert: {full_key} = {vdisp} outside {rng}"
                self._active_bounds_alerts[full_key] = msg
            else:
                # Observed a non-violating value → clear any persisted alert
                if full_key in self._active_bounds_alerts:
                    self._active_bounds_alerts.pop(full_key, None)

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

    # TODO: clean this up
    def _render_table(self, data: Dict[str, Any]) -> None:
        def _get_sort_key(namespace: str, subkey: str, key_priority: Iterable[str]) -> Tuple[int, object]:
            """Compute a stable sort key honoring an explicit key priority list.

            Returns (0, priority_index) when full_key in key_priority; otherwise
            (1, subkey.lower()) so prioritized keys appear first in given order.
            """
            full_key = f"{namespace}/{subkey}" if subkey else namespace
            try:
                priority_index = list(key_priority).index(full_key)
                return (0, priority_index)
            except ValueError:
                return (1, subkey.lower())

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
        charts: Dict[str, Dict[str, str]] = {}
        alerts: Dict[str, Dict[str, str]] = {}
        val_candidates: List[str] = []
        alert_candidates: List[str] = []
        key_candidates: List[str] = [ns + "/" for ns in ns_order]
        active_alerts = self.metrics_monitor.get_active_alerts()

        def _normalize_alert(msg: str) -> str:
            stripped = msg.strip()
            if stripped.startswith("⚠️"):
                stripped = stripped.replace("⚠️", "", 1).lstrip()
            return stripped

        alert_text_by_key: Dict[str, str] = {}
        for full_key, messages in active_alerts.items():
            normalized = [_normalize_alert(m) for m in messages if m]
            if normalized:
                alert_text_by_key[full_key] = " | ".join(normalized)
        for full_key, message in self._active_bounds_alerts.items():
            normalized = _normalize_alert(message)
            if not normalized:
                continue
            if full_key in alert_text_by_key and alert_text_by_key[full_key]:
                alert_text_by_key[full_key] = f"{alert_text_by_key[full_key]} | {normalized}"
            else:
                alert_text_by_key[full_key] = normalized

        for ns in ns_order:
            subdict = grouped[ns]
            f_sub: Dict[str, str] = {}
            c_sub: Dict[str, str] = {}
            a_sub: Dict[str, str] = {}
            for sub, v in subdict.items():
                # Add the subkey to the key candidates
                key_candidates.append(sub)
                full_key = f"{ns}/{sub}" if sub else ns
                precision = self.metric_precision_map.get(sub, 2)

                # Format the value
                val_str = number_to_string(
                    v,
                    precision=precision,
                    humanize=True,
                )

                # Add the highlight to the value
                if sub in self.highlight_value_bold_for_set:
                    val_str = _ansi(val_str, "bold", enable=self.color)

                # Add the delta to the value
                delta_str, color_name = self._delta_for_key(ns, sub, v)
                if delta_str:
                    delta_disp = _ansi(delta_str, color_name, enable=self.color)
                    val_disp = f"{val_str} {delta_disp}"
                else:
                    val_disp = val_str

                # Prepare a sparkline chart separately (fixed-width column)
                chart = ""
                if self.show_sparklines and is_number(v):
                    chart = self._spark_for_key(full_key, self.sparkline_width)

                # Add the subkey to the formatted data
                f_sub[sub] = val_disp
                c_sub[sub] = chart
                alert_plain = alert_text_by_key.get(full_key, "")
                if alert_plain:
                    alert_disp = _ansi(f"⚠️  {alert_plain}", "yellow", enable=self.color)
                else:
                    alert_disp = ""
                a_sub[sub] = alert_disp
                val_candidates.append(_strip_ansi(val_disp))
                if alert_disp:
                    alert_candidates.append(_strip_ansi(alert_disp))
            formatted[ns] = f_sub
            charts[ns] = c_sub
            alerts[ns] = a_sub

        # Add the border to the lines
        indent = self.indent
        key_width = max((len(k) for k in key_candidates), default=0)
        val_width = max((len(v) for v in val_candidates), default=0)
        val_width = max(val_width, self.min_val_width)
        alert_width = max((len(v) for v in alert_candidates), default=0)

        # Ensure the total table width does not shrink below min_table_width.
        # New layout with a fixed-width charts column:
        # "| " + (indent + key_width) + " | " + (val_width) + " | " + (chart_col_width) + " | " + (alert_col_width) + " |"
        static_cols = 2 + (indent + key_width) + 3 + 3 + 3 + 2
        if static_cols + val_width + self.chart_col_width + alert_width < self.min_table_width:
            val_width = self.min_table_width - static_cols - self.chart_col_width - alert_width

        border_len = 2 + (indent + key_width) + 3 + val_width + 3 + self.chart_col_width + 3 + alert_width + 2
        border = "-" * border_len
        lines: List[str] = []
        lines.append("")  # spacer

        for ns in ns_order:
            # Skip if the namespace is not in the formatted data
            if not formatted.get(ns): continue

            # Add the header to the lines
            header = ns + "/"
            alert_header = "alert" if alert_width else ""
            header_line = (
                f"| {header:<{indent + key_width}} | "
                f"{'':>{val_width}} | "
                f"{'':<{self.chart_col_width}} | "
                f"{alert_header:<{alert_width}} |"
            )
            lines.append(_ansi(header_line, "bold", enable=self.color))

            # Add the subkeys to the lines
            for sub, val in formatted[ns].items():
                val_display_len = len(_strip_ansi(val))
                val_padding = val_width - val_display_len
                val_padded = (" " * val_padding + val) if val_padding > 0 else val
                chart_str = charts.get(ns, {}).get(sub, "")
                # Truncate or pad chart to fixed width
                chart_clean = chart_str[: self.chart_col_width]
                chart_padded = f"{chart_clean:<{self.chart_col_width}}"
                alert_str = alerts.get(ns, {}).get(sub, "")
                alert_len = len(_strip_ansi(alert_str))
                alert_padding = alert_width - alert_len
                alert_padded = alert_str + (" " * alert_padding if alert_padding > 0 else "")

                # Add the key to the lines
                key_cell = f"{sub:<{key_width}}"
                highlight = False
                row_bg_color = None
                full_key = f"{ns}/{sub}" if sub else ns
                alert_active = full_key in active_alerts or full_key in self._active_bounds_alerts

                # Priority 1: trigger-based highlight (yellow) if alert is active for this key
                if alert_active:
                    highlight = True
                    row_bg_color = self.highlight_bounds_bg_color

                # Priority 2: bounds-based highlight (yellow); allow bare-name lookup
                bounds = self.metric_bounds_map.get(full_key) or self.metric_bounds_map.get(sub)
                if not highlight and bounds:
                    raw_val = grouped.get(ns, {}).get(sub)

                    # Skip if the value is not a number
                    if is_number(raw_val):
                        vnum = float(raw_val)

                        # Check if the value is below or above the bounds
                        below = ("min" in bounds) and (vnum < float(bounds["min"]))
                        above = ("max" in bounds) and (vnum > float(bounds["max"]))
                        if below or above:
                            # Set the highlight to true
                            highlight = True
                            row_bg_color = self.highlight_bounds_bg_color
                            
                # Priority 3: configured row highlight (blue)
                # Set the highlight to true if the subkey is in the highlight row for set
                if not highlight and sub in self.highlight_row_for_set:
                    if self.highlight_row_bold:
                        key_cell = _ansi(key_cell, "bold", enable=self.color)

                    # Set the highlight to true
                    highlight = True
                    row_bg_color = self.highlight_row_bg_color

                # Add the row to the lines
                row = (
                    f"| {' ' * indent}{key_cell} | {val_padded} | {chart_padded} | "
                    f"{alert_padded} |"
                )

                # Add the highlight to the row
                if highlight:
                    enable_bg = self.color or alert_active
                    row = _apply_bg(
                        row,
                        row_bg_color or self.highlight_row_bg_color,
                        enable=enable_bg,
                    )
                    if alert_active and not enable_bg:
                        # Provide a visible textual cue when colors are fully disabled
                        row = f"⚠️  {row}"
                lines.append(row)

        # Add the border to the lines
        lines.append(border)
        self._render_lines(lines)

        # Set the previous metrics
        self._prev = dict(data)
        self._last_height = len(lines)
