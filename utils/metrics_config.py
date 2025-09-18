"""Metrics configuration utilities (singleton-backed).

Loads and parses `config/metrics.yaml` exactly once and exposes helpers to
query derived views (precision, delta rules, bounds, highlights, etc.).
Free functions are kept for backward compatibility and now delegate to the
singleton so callers don't repeatedly read the file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from utils.io import read_yaml


@dataclass
class MetricsConfig:
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
        data = read_yaml(metrics_config_path) or {}
        self._config = data

    def _get_global_cfg(self) -> Dict[str, Any]:
        return self._config["_global"]

    def _get_metrics(self) -> Dict[str, Any]:
        metrics = [(name, value) for name, value in self._config.items() if not name.startswith("_") and isinstance(value, dict)]
        return metrics

    def metric_precision_dict(self) -> Dict[str, int]:
        """Convert metrics config to precision dict keyed by metric name. """
        precision_dict: Dict[str, int] = {}
        for metric_name, metric_config in self._get_metrics():
            precision = int(metric_config.get("precision", 2))
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
            if delta_rule == ">=": rule_fn = lambda prev, curr: curr >= prev
            else: continue

            # Add the rule function to the delta rules dictionary
            delta_rules[metric_name] = rule_fn

        # Return the delta rules dictionary
        return delta_rules

    # Note: algorithm-specific metric rules were removed.

    def key_priority(self) -> Optional[list]:
        """Preferred key ordering from metrics config (_global.key_priority)."""
        global_cfg = self._get_global_cfg()
        key_priority = global_cfg["key_priority"]
        dupes = [x for x in set(key_priority) if key_priority.count(x) > 1]
        assert len(dupes) == 0, f"key_priority must be a list of unique values, found duplicates: {dupes}"
        return key_priority

    def highlight_config(self) -> Dict[str, Any]:
        """Return highlight configuration for metrics table from metrics.yaml."""
        global_cfg = self._get_global_cfg() 
        highlight_cfg = global_cfg["highlight"]
        return highlight_cfg

    def metric_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return min/max bounds per metric if defined in metrics.yaml.

        Shape: {metric_name or namespaced: {"min": float, "max": float}}
        Missing bounds are omitted per metric.
        """
        bounds: Dict[str, Dict[str, float]] = {}
        for metric_name, metric_cfg in self._config.items():
            _bounds: Dict[str, float] = {}
            if "min" in metric_cfg: _bounds["min"] = float(metric_cfg["min"])
            if "max" in metric_cfg: _bounds["max"] = float(metric_cfg["max"])
            if _bounds: bounds[metric_name] = dict(_bounds)
        return bounds

metrics_config = MetricsConfig()
