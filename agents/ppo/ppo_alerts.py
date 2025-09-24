from utils.metric_bundles import MetricMonitorBundle
from utils.metrics_monitor import MetricAlert

# TODO: CLEAN this up

class PPOAlerts(MetricMonitorBundle):
    """PPO-specific metric alert bundle.

    Extends the shared alerts with PPO-centric tripwires for KL, clipping,
    and value-function diagnostics.
    """

    DEFAULT_SMOOTHING_WINDOW = 5

    def __init__(self, agent, *, smoothing_window: int | None = None) -> None:
        self.agent = agent
        if smoothing_window is None:
            smoothing_window = self.DEFAULT_SMOOTHING_WINDOW
        self._smoothing_window = max(1, int(smoothing_window))

    def _smoothed_metric_value(self, history: dict, metric_key: str) -> float | None:
        return self._windowed_metric_mean(
            history,
            metric_key,
            window=self._smoothing_window,
        )

    def _format_smoothed_value(self, history: dict, metric_key: str, value: float) -> str:
        avg_fmt = self._format_metric_value(metric_key, value)
        window_label = f"{self._smoothing_window}-step avg"
        latest_value = self._latest_metric_value(history, metric_key)
        if latest_value is None:
            return f"{window_label} {avg_fmt}"
        try:
            latest_fmt = self._format_metric_value(metric_key, float(latest_value))
        except (TypeError, ValueError):
            return f"{window_label} {avg_fmt}"
        return f"{window_label} {avg_fmt} (latest {latest_fmt})"

    def _monitor_approx_kl_oob(self, history: dict):
        metric_key = "train/approx_kl"
        value = self._smoothed_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        min_threshold, max_threshold = 1e-3, 5e-2
        smoothed_fmt = self._format_smoothed_value(history, metric_key, current)
        if current < min_threshold:
            threshold_fmt = self._format_metric_value(metric_key, min_threshold)
            return MetricAlert(
                _id=f"{metric_key}/oob_min",
                metric=metric_key,
                message=f"{smoothed_fmt} < {threshold_fmt}; updates may be too weak",
                tip="Increase the policy learning rate or relax the clip range.",
            )

        if current > max_threshold:
            threshold_fmt = self._format_metric_value(metric_key, max_threshold)
            return MetricAlert(
                _id=f"{metric_key}/oob_max",
                metric=metric_key,
                message=f"{smoothed_fmt} > {threshold_fmt}; updates may be too aggressive",
                tip="Decrease the policy learning rate or tighten the clip range.",
            )

    def _monitor_clip_fraction_oob(self, history: dict):
        metric_key = "train/clip_fraction"
        value = self._smoothed_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        min_threshold, max_threshold = 0.05, 0.5
        smoothed_fmt = self._format_smoothed_value(history, metric_key, current)
        if current < min_threshold:
            threshold_fmt = self._format_metric_value(metric_key, min_threshold)
            return MetricAlert(
                _id=f"{metric_key}/oob_min",
                metric=metric_key,
                message=f"{smoothed_fmt} < {threshold_fmt}; likely under-updating",
                tip="Increase the learning rate or decrease the clip range.",
            )

        if current > max_threshold:
            threshold_fmt = self._format_metric_value(metric_key, max_threshold)
            return MetricAlert(
                _id=f"{metric_key}/oob_max",
                metric=metric_key,
                message=f"{smoothed_fmt} > {threshold_fmt}; many updates are clipped",
                tip="Decrease the learning rate or increase the clip range.",
            )

    def _monitor_explained_variance_instability(self, history: dict):
        metric_key = "train/explained_variance"
        value = self._smoothed_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        low_threshold = -0.2
        high_threshold = 1.05
        smoothed_fmt = self._format_smoothed_value(history, metric_key, current)

        if current < low_threshold:
            return MetricAlert(
                _id=f"{metric_key}/too_low",
                metric=metric_key,
                message=f"{smoothed_fmt} indicates the value function is underfitting",
                tip="Increase value loss capacity, tune learning rates, or revisit returns normalization.",
            )

        if current > high_threshold:
            high_fmt = self._format_metric_value(metric_key, high_threshold)
            return MetricAlert(
                _id=f"{metric_key}/too_high",
                metric=metric_key,
                message=f"{smoothed_fmt} exceeds the stable range (> {high_fmt})",
                tip="Check for value leakage or normalize returns/advantages more aggressively.",
            )

    def _monitor_explained_var_worse_than_mean(self, history: dict):
        """Warn when value head explains less than the mean baseline.

        The `explained_var` metric is computed as 1 - Var(target - pred) / Var(target).
        Values < 0 mean the predictor performs worse than predicting the mean of
        the targets. We smooth over a short window to reduce noise and trigger a
        warning when the average is below 0.0.
        """
        metric_key = "train/explained_var"
        value = self._smoothed_metric_value(history, metric_key)
        if value is None:
            return None
        try:
            current = float(value)
        except (TypeError, ValueError):
            return None

        threshold = 0.0
        if current < threshold:
            smoothed_fmt = self._format_smoothed_value(history, metric_key, current)
            threshold_fmt = self._format_metric_value(metric_key, threshold)
            return MetricAlert(
                _id=f"{metric_key}/worse_than_mean",
                metric=metric_key,
                message=f"{smoothed_fmt} < {threshold_fmt}; predicting worse than mean",
                tip="Strengthen value learning (increase vf_coef or value lr) and verify returns normalization.",
            )

        
