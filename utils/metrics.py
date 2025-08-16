"""Metrics configuration utilities.

These helpers load the metrics.yaml and provide precision and validation
rules used by printing/logging callbacks. Kept separate from experiment
Config to avoid mixing concerns.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_metrics_config(config_dir: str = "config") -> Dict[str, Any]:
    """Load metrics configuration from YAML file.

    Args:
        config_dir: Directory containing the metrics.yaml file

    Returns:
        Dictionary containing metrics configuration
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    metrics_config_path = project_root / config_dir / "metrics.yaml"

    if not metrics_config_path.exists():
        raise FileNotFoundError(f"Metrics config file not found: {metrics_config_path}")

    with open(metrics_config_path, 'r') as f:
        return yaml.safe_load(f)


def get_metric_precision_dict(metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Convert metrics config to precision dictionary for StdoutMetricsTable.

    Expands metric names without namespaces to include common namespaces
    (train/, eval/, rollout/, time/).
    """
    if metrics_config is None:
        metrics_config = load_metrics_config()

    # Get default precision from global config
    default_precision = metrics_config.get('_global', {}).get('default_precision', 4)

    # Common namespaces where metrics can appear
    namespaces = ['train', 'eval', 'rollout', 'time']

    precision_dict: Dict[str, int] = {}

    # Process each metric in the new metric-centric structure
    for metric_name, metric_config in metrics_config.items():
        # Skip special entries like _global
        if metric_name.startswith('_') or not isinstance(metric_config, dict):
            continue

        # Get precision for this metric, default to global default
        precision = int(metric_config.get('precision', default_precision))

        # If force_integer is True, precision should be 0
        if metric_config.get('force_integer', False):
            precision = 0

        # Add the metric without namespace (for backward compatibility)
        precision_dict[metric_name] = precision

        # Add the metric with each namespace
        for namespace in namespaces:
            full_metric_name = f"{namespace}/{metric_name}"
            precision_dict[full_metric_name] = precision

    return precision_dict


def get_metric_delta_rules(metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, callable]:
    """Convert metrics config delta rules to callables for StdoutMetricsTable."""
    if metrics_config is None:
        metrics_config = load_metrics_config()

    namespaces = ['train', 'eval', 'rollout', 'time']
    delta_rules: Dict[str, callable] = {}

    # Process each metric in the new metric-centric structure
    for metric_name, metric_config in metrics_config.items():
        # Skip special entries like _global
        if metric_name.startswith('_') or not isinstance(metric_config, dict):
            continue

        # Check if this metric has a delta rule
        delta_rule = metric_config.get('delta_rule')
        if not delta_rule:
            continue

        if delta_rule == "non_decreasing":
            rule_fn = lambda prev, curr: curr >= prev
        else:
            # Add other rule types as needed
            continue

        # Add the rule for the metric without namespace
        delta_rules[metric_name] = rule_fn

        # Add the rule for the metric with each namespace
        for namespace in namespaces:
            full_metric_name = f"{namespace}/{metric_name}"
            delta_rules[full_metric_name] = rule_fn

    return delta_rules


def get_algorithm_metric_rules(algo_id: str, metrics_config: Optional[Dict[str, Any]] = None) -> Dict[str, dict]:
    """Get algorithm-specific metric validation rules."""
    if metrics_config is None:
        metrics_config = load_metrics_config()

    rules: Dict[str, dict] = {}
    namespaces = ['train', 'eval', 'rollout', 'time']

    # Process each metric in the new metric-centric structure
    for metric_name, metric_config in metrics_config.items():
        # Skip special entries like _global
        if metric_name.startswith('_') or not isinstance(metric_config, dict):
            continue

        # Check if this metric has algorithm-specific rules
        algorithm_rules = metric_config.get('algorithm_rules', {})
        if not algorithm_rules:
            continue

        # Check if there's a rule for this specific algorithm
        rule_config = algorithm_rules.get(algo_id.lower())
        if not rule_config:
            continue

        threshold = rule_config.get('threshold')
        condition = rule_config.get('condition')
        message = rule_config.get('message', 'Metric validation failed')
        level = rule_config.get('level', 'warning')

        # Create the validation function based on condition
        if condition == "less_than":
            check_fn = lambda value: value < threshold
        elif condition == "greater_than":
            check_fn = lambda value: value > threshold
        elif condition == "between":
            min_val = rule_config.get('min', float('-inf'))
            max_val = rule_config.get('max', float('inf'))
            check_fn = lambda value: min_val <= value <= max_val
        else:
            continue

        rule_dict = {
            'check': check_fn,
            'message': message,
            'level': level
        }

        # Add rules for metric with each namespace
        for namespace in namespaces:
            full_metric_name = f"{namespace}/{metric_name}"
            rules[full_metric_name] = rule_dict

    return rules


def get_key_priority(metrics_config: Optional[Dict[str, Any]] = None) -> Optional[list]:
    """Return preferred key ordering from metrics config (_global.key_priority) if available.

    Args:
        metrics_config: Optional preloaded metrics config dict

    Returns:
        A list of metric keys in preferred order, or None if not configured.
    """
    if metrics_config is None:
        try:
            metrics_config = load_metrics_config()
        except Exception:
            return None

    global_cfg = metrics_config.get('_global', {}) if isinstance(metrics_config, dict) else {}
    kp = global_cfg.get('key_priority')
    # Ensure it's a list of strings
    if isinstance(kp, list) and all(isinstance(x, str) for x in kp):
        return kp
    return None
