"""Metrics configuration utilities (singleton-backed).

Loads and parses `config/metrics.yaml` exactly once and exposes helpers to
query derived views (precision, delta rules, bounds, highlights, etc.).
Free functions are kept for backward compatibility and now delegate to the
singleton so callers don't repeatedly read the file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import yaml

@dataclass
class Metrics:
    """Singleton-style accessor for metrics.yaml-derived data.

    The underlying YAML file is loaded once at initialization. All query
    methods operate on the cached dictionary. The file is assumed immutable
    for the lifetime of the process.
    """

    config_dir: str = "config"
    _config: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._load()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def _load(self) -> None:
        project_root = Path(__file__).parent.parent
        metrics_config_path = project_root / self.config_dir / "metrics.yaml"
        with open(metrics_config_path, "r") as f: data = yaml.safe_load(f)
        self._config = data

    def _get_global_cfg(self) -> Dict[str, Any]:
        return self._config["_global"]

    def _get_metrics(self) -> Dict[str, Any]:
        global_cfg = self._get_global_cfg()
        metrics = [(name, value) for name, value in global_cfg.items() if not name.startswith("_") and isinstance(value, dict)]
        return metrics

    def get_default_precision(self) -> int:
        global_cfg = self._get_global_cfg()
        default_precision = global_cfg["default_precision"]
        return default_precision

    def metric_precision_dict(self) -> Dict[str, int]:
        """Convert metrics config to precision dict keyed by metric name. """
        default_precision = self.get_default_precision()
        precision_dict: Dict[str, int] = {}
        for metric_name, metric_config in self._get_metrics():
            precision = int(metric_config.get("precision", default_precision))
            precision_dict[metric_name] = precision
        return precision_dict

    def metric_delta_rules(self) -> Dict[str, Callable]:
        """Return delta validation rules per metric (as callables)."""
        delta_rules: Dict[str, Callable] = {}

        # Iterate over all metrics and add the delta rule to the delta rules dictionary
        for metric_name, metric_config in self._get_metrics():
            # If no delta rule is defined, skip
            delta_rule = metric_config.get("delta_rule")
            if not delta_rule: continue

            # If delta rule is non-decreasing, set the rule function
            if delta_rule == "non_decreasing": rule_fn = lambda prev, curr: curr >= prev
            else: continue

            # Add the rule function to the delta rules dictionary
            delta_rules[metric_name] = rule_fn

        # Return the delta rules dictionary
        return delta_rules

    def algorithm_metric_rules(self, algo_id: str) -> Dict[str, dict]:
        """Return algorithm-specific metric validation rules."""
        rules: Dict[str, dict] = {}
        namespaces = ["train", "eval", "rollout", "time"]

        for metric_name, metric_config in self._get_metrics():
            # If no algorithm rules are defined, skip
            algorithm_rules = metric_config.get("algorithm_rules", {})
            rule_config = algorithm_rules.get(algo_id.lower())
            if not rule_config: continue

            threshold = rule_config.get("threshold")
            condition = rule_config.get("condition")
            message = rule_config.get("message", "Metric validation failed")
            level = rule_config.get("level", "warning")

            if condition == "less_than":
                check_fn = lambda value: value < threshold
            elif condition == "greater_than":
                check_fn = lambda value: value > threshold
            elif condition == "between":
                min_val = rule_config.get("min", float("-inf"))
                max_val = rule_config.get("max", float("inf"))
                check_fn = lambda value: min_val <= value <= max_val
            else:
                continue

            rule_dict = {"check": check_fn, "message": message, "level": level}
            for namespace in namespaces:
                full_metric_name = f"{namespace}/{metric_name}"
                rules[full_metric_name] = rule_dict

        return rules

    def key_priority(self) -> Optional[list]:
        """Preferred key ordering from metrics config (_global.key_priority)."""
        global_cfg = self._get_global_cfg()
        key_priority = global_cfg["key_priority"]
        return key_priority

    def highlight_config(self) -> Dict[str, Any]:
        """Return highlight configuration for metrics table from metrics.yaml."""
        hl = self._get_global_cfg()["highlight"]

        default_row_metrics = {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "total_timesteps"}
        default_value_bold_metrics = {"ep_rew_mean", "ep_rew_last", "ep_rew_best", "epoch"}

        def _as_set(x, default):
            if isinstance(x, list):
                return {str(v) for v in x}
            return set(default)

        row_metrics = _as_set(hl.get("row_metrics"), default_row_metrics)
        value_bold_metrics = _as_set(hl.get("value_bold_metrics"), default_value_bold_metrics)
        row_bg_color = str(hl.get("row_bg_color", "bg_blue"))
        row_bold = bool(hl.get("row_bold", True))

        return {
            "row_metrics": row_metrics,
            "value_bold_metrics": value_bold_metrics,
            "row_bg_color": row_bg_color,
            "row_bold": row_bold,
        }

    def metric_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return min/max bounds per metric if defined in metrics.yaml.

        Shape: {metric_name or namespaced: {"min": float, "max": float}}
        Missing bounds are omitted per metric.
        """
        bounds: Dict[str, Dict[str, float]] = {}

        for metric_name, metric_cfg in self._config.items():
            # Skip private metrics and non-dict entries
            if metric_name.startswith("_") or not isinstance(metric_cfg, dict): continue

            # Initialize bounds dict for the metric
            _bounds: Dict[str, float] = {}
            if "min" in metric_cfg: _bounds["min"] = float(metric_cfg["min"])
            if "max" in metric_cfg: _bounds["max"] = float(metric_cfg["max"])
            if _bounds: bounds[metric_name] = dict(_bounds)

        return bounds


metrics_config = Metrics()
