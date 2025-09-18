from utils.metric_bundles import MetricMonitorBundle
from utils.metrics_config import metrics_config
from utils.metrics_monitor import MetricAlert


class PPOAlerts(MetricMonitorBundle):
    """PPO-specific metric alert bundle.

    Extends the shared alerts with PPO-centric tripwires for KL, clipping,
    and value-function diagnostics.
    """

    def __init__(self, agent) -> None:
        self.agent = agent

    def _latest_metric_value(self, history: dict, metric_key: str):
        series = history.get(metric_key)
        if not series:
            return None
        _, value = series[-1]
        return value

    def _format_metric_value(self, metric_key: str, value: float) -> str:
        precision = metrics_config.precision_for_metric(metric_key)
        format_str = f"{{:.{precision}f}}"
        return format_str.format(value)

    def _monitor_approx_kl_oob(self, history: dict):
        metric_key = "train/approx_kl"
        value = self._latest_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        min_threshold, max_threshold = 1e-3, 5e-2
        if current < min_threshold:
            threshold_fmt = self._format_metric_value(metric_key, min_threshold)
            curr_fmt = self._format_metric_value(metric_key, current)
            return MetricAlert(
                _id=f"{metric_key}/oob_min",
                metric=metric_key,
                message=f"{curr_fmt} < {threshold_fmt}; updates may be too weak",
                tip="Increase the policy learning rate or relax the clip range.",
            )

        if current > max_threshold:
            threshold_fmt = self._format_metric_value(metric_key, max_threshold)
            curr_fmt = self._format_metric_value(metric_key, current)
            return MetricAlert(
                _id=f"{metric_key}/oob_max",
                metric=metric_key,
                message=f"{curr_fmt} > {threshold_fmt}; updates may be too aggressive",
                tip="Decrease the policy learning rate or tighten the clip range.",
            )

    def _monitor_clip_fraction_oob(self, history: dict):
        metric_key = "train/clip_fraction"
        value = self._latest_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        min_threshold, max_threshold = 0.05, 0.5
        if current < min_threshold:
            threshold_fmt = self._format_metric_value(metric_key, min_threshold)
            curr_fmt = self._format_metric_value(metric_key, current)
            return MetricAlert(
                _id=f"{metric_key}/oob_min",
                metric=metric_key,
                message=f"{curr_fmt} < {threshold_fmt}; likely under-updating",
                tip="Increase the learning rate or decrease the clip range.",
            )

        if current > max_threshold:
            threshold_fmt = self._format_metric_value(metric_key, max_threshold)
            curr_fmt = self._format_metric_value(metric_key, current)
            return MetricAlert(
                _id=f"{metric_key}/oob_max",
                metric=metric_key,
                message=f"{curr_fmt} > {threshold_fmt}; many updates are clipped",
                tip="Decrease the learning rate or increase the clip range.",
            )

    def _monitor_explained_variance_instability(self, history: dict):
        metric_key = "train/explained_variance"
        value = self._latest_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        low_threshold = -0.2
        high_threshold = 1.05
        curr_fmt = self._format_metric_value(metric_key, current)

        if current < low_threshold:
            return MetricAlert(
                _id=f"{metric_key}/too_low",
                metric=metric_key,
                message=f"{curr_fmt} indicates the value function is underfitting",
                tip="Increase value loss capacity, tune learning rates, or revisit returns normalization.",
            )

        if current > high_threshold:
            high_fmt = self._format_metric_value(metric_key, high_threshold)
            return MetricAlert(
                _id=f"{metric_key}/too_high",
                metric=metric_key,
                message=f"{curr_fmt} exceeds the stable range (> {high_fmt})",
                tip="Check for value leakage or normalize returns/advantages more aggressively.",
            )
