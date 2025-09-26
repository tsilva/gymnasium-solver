from utils.metric_bundles import MetricMonitorBundle
from utils.metrics_monitor import MetricAlert

# TODO: CLEAN this up

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
        metric_key = "train/opt/ppo/approx_kl"
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
        metric_key = "train/opt/ppo/clip_fraction"
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

    def _monitor_kl_vs_approx_kl_divergence(self, history: dict):
        kl_key = "train/opt/ppo/kl"
        approx_key = "train/opt/ppo/approx_kl"
        kl_value = self._smoothed_metric_value(history, kl_key)
        approx_value = self._smoothed_metric_value(history, approx_key)
        if kl_value is None or approx_value is None:
            return None

        try:
            kl_current = float(kl_value)
            approx_current = float(approx_value)
        except (TypeError, ValueError):
            return None

        kl_mag = abs(kl_current)
        approx_mag = abs(approx_current)
        if max(kl_mag, approx_mag) < _MIN_KL_MAGNITUDE:
            return None

        alerts: list[MetricAlert] = []

        if approx_mag > 0.0:
            ratio = kl_mag / approx_mag
            if ratio >= _KL_RATIO_THRESHOLD:
                kl_fmt = self._format_smoothed_value(history, kl_key, kl_current)
                approx_fmt = self._format_smoothed_value(history, approx_key, approx_current)
                ratio_fmt = f"{ratio:.1f}x"
                alerts.append(
                    MetricAlert(
                        _id=f"{kl_key}/dominates_approx_kl",
                        metric=kl_key,
                        message=(
                            f"{kl_fmt} vs {approx_fmt} (~{ratio_fmt}); updates may be too aggressive; "
                            "clipping may not fully constrain the step."
                        ),
                        tip=(
                            "Reduce the policy step size by lowering the learning rate, running fewer epochs per "
                            "rollout, or shrinking the clip range."
                        ),
                    )
                )

        if kl_mag > 0.0:
            inverse_ratio = approx_mag / kl_mag
            if inverse_ratio >= _KL_RATIO_THRESHOLD:
                approx_fmt = self._format_smoothed_value(history, approx_key, approx_current)
                kl_fmt = self._format_smoothed_value(history, kl_key, kl_current)
                ratio_fmt = f"{inverse_ratio:.1f}x"
                alerts.append(
                    MetricAlert(
                        _id=f"{approx_key}/dominates_true_kl",
                        metric=approx_key,
                        message=(
                            f"{approx_fmt} vs {kl_fmt} (~{ratio_fmt}); surrogate KL far exceeds measured KL; "
                            "early-stop heuristics may fire too early."
                        ),
                        tip=(
                            "Consider loosening clip_range, recomputing KL on fresh rollouts, or basing early stops "
                            "on the measured KL instead of the surrogate."
                        ),
                    )
                )

        return alerts or None

    def _monitor_explained_var_instability(self, history: dict):
        metric_key = "train/opt/value/explained_var"
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
        metric_key = "train/opt/value/explained_var"
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

        
