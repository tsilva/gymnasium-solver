"""Singleton-backed utilities for metrics.yaml precision, bounds, and keys."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.io import read_yaml

# Reused across helpers to avoid per-call allocations
_ALLOWED_NAMESPACES = frozenset({"train", "val", "test"})

@dataclass
class MetricsConfig:
    """Singleton accessor for metrics.yaml-derived precision, bounds, and rules."""

    config_dir: str = "config"
    _config: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._load()

    def _load(self) -> None:
        project_root = Path(__file__).parent.parent
        metrics_config_path = project_root / self.config_dir / "metrics.yaml"
        data = read_yaml(metrics_config_path) or {}
        self._config = data

    def _get_global_cfg(self) -> Dict[str, Any]:
        return self._config["_global"]

    @staticmethod
    def add_namespace_to_metric(namespace: str, subkey: Optional[str]) -> str:
        """Build a fully-qualified metric name validating the namespace. """
        assert namespace in _ALLOWED_NAMESPACES, f"Invalid metrics namespace '{namespace}'. Expected one of: {sorted(_ALLOWED_NAMESPACES)}"
        return f"{namespace}/{subkey}" if subkey else namespace

    @staticmethod
    def remove_namespace_from_metric(metric_name: str) -> str:
        """Extract the subkey from a fully-qualified metric name """
        assert MetricsConfig.is_namespaced_metric(metric_name), f"Invalid metric key '{metric_name}'"
        _, subkey = metric_name.split("/", 1)
        return subkey

    @staticmethod
    def is_namespaced_metric(metric_name: str) -> bool:
        """Return True for "<namespace>/<subkey>" where namespace in {train,val,test}."""
        # Assert key is not empty
        assert metric_name, "Invalid metric key: empty string"

        # Invalid if no slash in key
        if "/" not in metric_name: return False
        
        # Invalid if no subkey
        namespace, subkey = metric_name.split("/", 1)
        if not subkey: return False

        # Invalid if namespace is not in allowed namespaces
        if not namespace in _ALLOWED_NAMESPACES: return False

        # Valid if all checks pass
        return True
    
    @staticmethod
    def ensure_unnamespaced_metric(metric_name: str) -> str:
        """Ensure a metric name is unnamespaced and valid."""
        if not MetricsConfig.is_namespaced_metric(metric_name): return metric_name
        metric_name = MetricsConfig.remove_namespace_from_metric(metric_name)
        return metric_name

    def key_priority(self) -> Optional[list]:
        """Preferred key ordering from metrics config (_global.key_priority)."""
        global_cfg = self._get_global_cfg()
        key_priority = global_cfg["key_priority"]
        dupes = [x for x in set(key_priority) if key_priority.count(x) > 1]
        assert len(dupes) == 0, f"key_priority must be a list of unique values, found duplicates: {dupes}"
        return key_priority

    def total_timesteps_key(self) -> str:
        """Return the canonical step metric key from metrics config."""
        global_cfg = self._get_global_cfg()
        key = global_cfg["step_key"]
        return key

    def get_metrics_bounds_violations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get violations of metrics within configured bounds."""
        # Collect out of bounds violations (if any)
        violations: List[str] = []
        for metric_name, value in metrics.items():
            # If no bounds, skip
            bounds = self.bounds_for_metric(metric_name)
            if not bounds: continue
            
            # If value is less than min bound, add violation
            bounds_min = bounds.get("min")
            if bounds_min and value < bounds_min: violations.append(f"{metric_name}={value} < min {bounds_min}")

            # If value is greater than max bound, add violation
            bounds_max = bounds.get("max")
            if bounds_max and value > bounds_max: violations.append(f"{metric_name}={value} > max {bounds_max}")

        return violations

    def bounds_for_metric(self, metric_name: str) -> Dict[str, float]:
        metric_name = self.ensure_unnamespaced_metric(metric_name)
        metric_cfg = self._config.get(metric_name, {})
        bounds = metric_cfg.get("bounds")
        if bounds:
            return bounds

        fallback: Dict[str, float] = {}
        if "min" in metric_cfg:
            fallback["min"] = float(metric_cfg["min"])
        if "max" in metric_cfg:
            fallback["max"] = float(metric_cfg["max"])

        return fallback or None

    def delta_rules_for_metric(self, metric_name: str) -> Callable:
        metric_name = self.ensure_unnamespaced_metric(metric_name)
        delta_rule = self._config.get(metric_name, {}).get("delta_rule")
        if not delta_rule: return None
        if delta_rule == ">=": delta_rule = lambda prev, curr: curr >= prev
        else: raise ValueError(f"Invalid delta rule '{delta_rule}' for metric '{metric_name}'")
        return delta_rule

    def precision_for_metric(self, metric_name: str) -> int:
        metric_name = self.ensure_unnamespaced_metric(metric_name)
        precision = self._config.get(metric_name, {}).get("precision", 2)
        return precision

    def style_for_metric(self, metric_name: str) -> Dict[str, Any]:
        metric_name = self.ensure_unnamespaced_metric(metric_name)
        config = self._config.get(metric_name, {})
        highlight = config.get("highlight", False)
        bold = config.get("bold", False)
        style = {
            "highlight": highlight,
            "bold": bold,
        }
        return style

    def assert_metrics_within_bounds(self, metrics: Dict[str, Any]) -> None:
        bound_violations = self.get_metrics_bounds_violations(metrics)
        if bound_violations: raise ValueError("Out-of-bounds metrics: " + "; ".join(bound_violations))
    

metrics_config = MetricsConfig()
