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


# ------------------------------
# Singleton implementation
# ------------------------------

_METRICS_SINGLETON: "Metrics | None" = None


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
        if not metrics_config_path.exists():
            raise FileNotFoundError(
                f"Metrics config file not found: {metrics_config_path}"
            )
        with open(metrics_config_path, "r") as f:
            data = yaml.safe_load(f)
        # Ensure a dict even if file is empty
        self._config = data if isinstance(data, dict) else {}

    # --------- Query helpers (instance methods) ---------
    def metric_precision_dict(self) -> Dict[str, int]:
        """Convert metrics config to precision dict keyed by metric name.

        Expands bare metric names across common namespaces
        (train/, eval/, rollout/, time/).
        """
        metrics_config = self._config
        default_precision = metrics_config.get("_global", {}).get("default_precision", 4)
        namespaces = ["train", "eval", "rollout", "time"]

        precision_dict: Dict[str, int] = {}
        for metric_name, metric_config in metrics_config.items():
            if metric_name.startswith("_") or not isinstance(metric_config, dict):
                continue

            precision = int(metric_config.get("precision", default_precision))
            if metric_config.get("force_integer", False):
                precision = 0

            precision_dict[metric_name] = precision
            for namespace in namespaces:
                full_metric_name = f"{namespace}/{metric_name}"
                precision_dict[full_metric_name] = precision

        return precision_dict

    def metric_delta_rules(self) -> Dict[str, Callable]:
        """Return delta validation rules per metric (as callables)."""
        metrics_config = self._config
        namespaces = ["train", "eval", "rollout", "time"]
        delta_rules: Dict[str, Callable] = {}

        for metric_name, metric_config in metrics_config.items():
            if metric_name.startswith("_") or not isinstance(metric_config, dict):
                continue

            delta_rule = metric_config.get("delta_rule")
            if not delta_rule:
                continue

            if delta_rule == "non_decreasing":
                rule_fn = lambda prev, curr: curr >= prev
            else:
                # Add other rule types as needed
                continue

            delta_rules[metric_name] = rule_fn
            for namespace in namespaces:
                full_metric_name = f"{namespace}/{metric_name}"
                delta_rules[full_metric_name] = rule_fn

        return delta_rules

    def algorithm_metric_rules(self, algo_id: str) -> Dict[str, dict]:
        """Return algorithm-specific metric validation rules."""
        metrics_config = self._config
        rules: Dict[str, dict] = {}
        namespaces = ["train", "eval", "rollout", "time"]

        for metric_name, metric_config in metrics_config.items():
            if metric_name.startswith("_") or not isinstance(metric_config, dict):
                continue

            algorithm_rules = metric_config.get("algorithm_rules", {})
            if not algorithm_rules:
                continue

            rule_config = algorithm_rules.get(algo_id.lower())
            if not rule_config:
                continue

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
        metrics_config = self._config
        global_cfg = metrics_config.get("_global", {}) if isinstance(metrics_config, dict) else {}
        kp = global_cfg.get("key_priority")
        if isinstance(kp, list) and all(isinstance(x, str) for x in kp):
            return kp
        return None

    def highlight_config(self) -> Dict[str, Any]:
        """Return highlight configuration for metrics table from metrics.yaml."""
        metrics_config = self._config
        global_cfg = metrics_config.get("_global", {}) if isinstance(metrics_config, dict) else {}
        hl = global_cfg.get("highlight", {}) if isinstance(global_cfg, dict) else {}

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

        Expands into common namespaces (train/, val/, rollout/, time/).
        Shape: {metric_name or namespaced: {"min": float, "max": float}}
        Missing bounds are omitted per metric.
        """
        metrics_config = self._config
        namespaces = ["train", "val", "rollout", "time"]
        bounds: Dict[str, Dict[str, float]] = {}

        for metric_name, metric_cfg in metrics_config.items():
            if metric_name.startswith("_") or not isinstance(metric_cfg, dict):
                continue
            has_min = "min" in metric_cfg
            has_max = "max" in metric_cfg
            if not (has_min or has_max):
                continue

            b: Dict[str, float] = {}
            if has_min:
                try:
                    b["min"] = float(metric_cfg["min"])
                except Exception:
                    pass
            if has_max:
                try:
                    b["max"] = float(metric_cfg["max"])
                except Exception:
                    pass
            if not b:
                continue

            bounds[metric_name] = dict(b)
            for ns in namespaces:
                bounds[f"{ns}/{metric_name}"] = dict(b)

        return bounds


metrics_config = Metrics()
