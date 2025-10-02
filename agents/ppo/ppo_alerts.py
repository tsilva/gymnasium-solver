from utils.metric_bundles import MetricMonitorBundle
from utils.metrics_monitor import MetricAlert

DEFAULT_SMOOTHING_WINDOW = 5
_KL_RATIO_THRESHOLD = 2.0
_MIN_KL_MAGNITUDE = 1e-4

class PPOAlerts(MetricMonitorBundle):
    """PPO-specific metric alert bundle.

    Extends the shared alerts with PPO-centric tripwires for KL, clipping,
    and value-function diagnostics.
    """

    def __init__(self, agent, *, smoothing_window: int = DEFAULT_SMOOTHING_WINDOW) -> None:
        self.agent = agent
        self._smoothing_window = max(1, int(smoothing_window))

    def _smoothed_metric_value(self, history: dict, metric_key: str) -> float | None:
        return self._windowed_metric_mean(history, metric_key, window=self._smoothing_window)

    def _format_smoothed_value(self, history: dict, metric_key: str, value: float) -> str:
        avg_fmt = self._format_metric_value(metric_key, value)
        window_label = f"{self._smoothing_window}-step avg"
        latest_value = self._latest_metric_value(history, metric_key)
        if latest_value is None:
            return f"{window_label} {avg_fmt}"
        latest_fmt = self._format_metric_value(metric_key, float(latest_value))
        return f"{window_label} {avg_fmt} (latest {latest_fmt})"

    def _check_threshold_bounds(
        self, history: dict, metric_key: str, min_threshold: float, max_threshold: float,
        min_message: str, min_tip: str, max_message: str, max_tip: str
    ):
        value = self._smoothed_metric_value(history, metric_key)
        if value is None:
            return None
        current = float(value)
        smoothed_fmt = self._format_smoothed_value(history, metric_key, current)

        if current < min_threshold:
            threshold_fmt = self._format_metric_value(metric_key, min_threshold)
            return MetricAlert(
                _id=f"{metric_key}/oob_min",
                metric=metric_key,
                message=f"{smoothed_fmt} < {threshold_fmt}; {min_message}",
                tip=min_tip,
            )

        if current > max_threshold:
            threshold_fmt = self._format_metric_value(metric_key, max_threshold)
            return MetricAlert(
                _id=f"{metric_key}/oob_max",
                metric=metric_key,
                message=f"{smoothed_fmt} > {threshold_fmt}; {max_message}",
                tip=max_tip,
            )

    def _monitor_approx_kl_oob(self, history: dict):
        return self._check_threshold_bounds(
            history, "train/opt/ppo/approx_kl", 1e-3, 5e-2,
            "updates may be too weak",
            "Increase the policy learning rate or relax the clip range.",
            "updates may be too aggressive",
            "Decrease the policy learning rate or tighten the clip range.",
        )

    def _monitor_clip_fraction_oob(self, history: dict):
        return self._check_threshold_bounds(
            history, "train/opt/ppo/clip_fraction", 0.05, 0.5,
            "likely under-updating",
            "Increase the learning rate or decrease the clip range.",
            "many updates are clipped",
            "Decrease the learning rate or increase the clip range.",
        )

    def _check_kl_ratio(self, history: dict, num_key: str, denom_key: str, num_val: float, denom_val: float, alert_id: str, message_template: str, tip: str):
        if denom_val <= 0.0:
            return None
        ratio = abs(num_val) / abs(denom_val)
        if ratio < _KL_RATIO_THRESHOLD:
            return None
        num_fmt = self._format_smoothed_value(history, num_key, num_val)
        denom_fmt = self._format_smoothed_value(history, denom_key, denom_val)
        return MetricAlert(
            _id=alert_id,
            metric=num_key,
            message=f"{num_fmt} vs {denom_fmt} (~{ratio:.1f}x); {message_template}",
            tip=tip,
        )

    def _monitor_kl_vs_approx_kl_divergence(self, history: dict):
        kl_key = "train/opt/ppo/kl"
        approx_key = "train/opt/ppo/approx_kl"
        kl_value = self._smoothed_metric_value(history, kl_key)
        approx_value = self._smoothed_metric_value(history, approx_key)
        if kl_value is None or approx_value is None:
            return None

        kl_current = float(kl_value)
        approx_current = float(approx_value)
        if max(abs(kl_current), abs(approx_current)) < _MIN_KL_MAGNITUDE:
            return None

        alerts = []
        alert = self._check_kl_ratio(
            history, kl_key, approx_key, kl_current, approx_current,
            f"{kl_key}/dominates_approx_kl",
            "updates may be too aggressive; clipping may not fully constrain the step.",
            "Reduce the policy step size by lowering the learning rate, running fewer epochs per rollout, or shrinking the clip range.",
        )
        if alert:
            alerts.append(alert)

        alert = self._check_kl_ratio(
            history, approx_key, kl_key, approx_current, kl_current,
            f"{approx_key}/dominates_true_kl",
            "surrogate KL far exceeds measured KL; early-stop heuristics may fire too early.",
            "Consider loosening clip_range, recomputing KL on fresh rollouts, or basing early stops on the measured KL instead of the surrogate.",
        )
        if alert:
            alerts.append(alert)

        return alerts or None

    def _check_single_threshold(self, history: dict, metric_key: str, threshold: float, alert_id: str, message: str, tip: str, below: bool = True):
        value = self._smoothed_metric_value(history, metric_key)
        if value is None:
            return None
        current = float(value)
        if (below and current >= threshold) or (not below and current <= threshold):
            return None
        smoothed_fmt = self._format_smoothed_value(history, metric_key, current)
        threshold_fmt = self._format_metric_value(metric_key, threshold)
        operator = "<" if below else ">"
        return MetricAlert(
            _id=alert_id,
            metric=metric_key,
            message=f"{smoothed_fmt} {operator} {threshold_fmt}; {message}",
            tip=tip,
        )

    def _monitor_explained_var_instability(self, history: dict):
        metric_key = "train/opt/value/explained_var"
        alert = self._check_single_threshold(
            history, metric_key, -0.2, f"{metric_key}/too_low",
            "indicates the value function is underfitting",
            "Increase value loss capacity, tune learning rates, or revisit returns normalization.",
            below=True,
        )
        if alert:
            return alert
        return self._check_single_threshold(
            history, metric_key, 1.05, f"{metric_key}/too_high",
            "exceeds the stable range",
            "Check for value leakage or normalize returns/advantages more aggressively.",
            below=False,
        )

    def _monitor_explained_var_worse_than_mean(self, history: dict):
        """Warn when value head explains less than the mean baseline."""
        return self._check_single_threshold(
            history, "train/opt/value/explained_var", 0.0,
            "train/opt/value/explained_var/worse_than_mean",
            "predicting worse than mean",
            "Strengthen value learning (increase vf_coef or value lr) and verify returns normalization.",
            below=True,
        )
