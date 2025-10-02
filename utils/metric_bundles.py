from statistics import fmean
from typing import Callable, Iterable, List

import math

# Prefer Gymnasium spaces; fall back gracefully if not present in test stubs
from gymnasium import spaces as gym_spaces  # type: ignore

import pytorch_lightning as pl

from utils.metrics_config import metrics_config
from utils.metrics_monitor import MetricAlert

# TODO: REFACTOR this file

class MetricMonitorBundle:
    """Interface for metric monitor bundles.

    Implementations should return an iterable of monitor callables via
    `get_monitor_fns()`. Each callable receives the metrics history dict
    and returns either a `MetricAlert`, an iterable of `MetricAlert`, or
    a false-y value when no alert should be emitted.

    The default implementation auto-registers any bound method whose
    name starts with `_monitor`.
    """

    def get_monitor_fns(self) -> Iterable[Callable]:
        fns: list[Callable] = []
        for name in dir(self):
            if not name.startswith("_monitor"):
                continue
            fn = getattr(self, name)
            if not callable(fn):
                continue
            fns.append(fn)
        return tuple(fns)

    # ---- shared helpers ----
    def _metric_series(self, history: dict, metric_key: str):
        return history.get(metric_key) or []

    def _latest_metric_value(self, history: dict, metric_key: str):
        series = self._metric_series(history, metric_key)
        if not series:
            return None
        _, value = series[-1]
        return value

    def _windowed_metric_mean(
        self,
        history: dict,
        metric_key: str,
        *,
        window: int | None = None,
    ) -> float | None:
        series = self._metric_series(history, metric_key)
        if not series:
            return None

        if not window or window <= 0:
            window = len(series)

        values = [float(raw) for _, raw in series[-window:]]
        if not values:
            return None

        return fmean(values)

    def _format_metric_value(self, metric_key: str, value: float) -> str:
        precision = metrics_config.precision_for_metric(metric_key)
        format_str = f"{{:.{precision}f}}"
        return format_str.format(float(value))

    def _format_smoothed_with_latest(
        self,
        history: dict,
        metric_key: str,
        smoothed_value: float,
    ) -> str:
        """Format smoothed value with optional latest value suffix."""
        smoothed_fmt = self._format_metric_value(metric_key, smoothed_value)
        latest_val = self._latest_metric_value(history, metric_key)
        if latest_val is None:
            return smoothed_fmt
        latest_fmt = self._format_metric_value(metric_key, float(latest_val))
        return f"{smoothed_fmt} (latest {latest_fmt})"

    def _numeric_series(self, series: Iterable[tuple]) -> list[float]:
        return [float(raw) for _step, raw in series]


# TODO: call core
class CoreMetricAlerts(MetricMonitorBundle):
    """Shared metric tripwires that apply to all algorithms."""

    _BOUNDS_SMOOTHING_WINDOW = 5
    _EP_REWARD_WINDOW = 8
    _EP_REWARD_STALL_DELTA_RATIO = 0.01
    _EP_REWARD_STALL_DELTA_MIN = 0.1
    _EP_REWARD_DECLINE_RATIO = 0.03
    _EP_REWARD_DECLINE_MIN = 0.3

    def __init__(self, pl_module: pl.LightningModule | None = None) -> None:
        self.pl_module = pl_module
        self._step_key = metrics_config.total_timesteps_key()

    # ---- monitors ----
    def _monitor_nan_metrics(self, history: dict):
        alerts: List[MetricAlert] = []
        for metric_key, series in history.items():
            if not series:
                continue
            _step, value = series[-1]
            value_float = float(value)

            if not (math.isnan(value_float) or math.isinf(value_float)):
                continue

            alerts.append(
                MetricAlert(
                    _id=f"{metric_key}/nan_or_inf",
                    metric=metric_key,
                    message="latest value is NaN/Inf",
                    tip="Check gradients, reward scaling, or numerical stability to restore finite metrics.",
                )
            )

        return alerts or None

    def _episode_reward_metric_keys(self, history: dict) -> Iterable[str]:
        for metric_key in history.keys():
            if metric_key.endswith("roll/ep_rew/mean"):
                yield metric_key

    def _monitor_step_progress(self, history: dict):
        step_key = self._step_key
        series = history.get(step_key)
        if not series or len(series) < 2:
            return None
        _, prev_value = series[-2]
        _, current_value = series[-1]
        if current_value <= prev_value:
            prev_fmt = self._format_metric_value(step_key, float(prev_value))
            curr_fmt = self._format_metric_value(step_key, float(current_value))
            return MetricAlert(
                _id=f"{step_key}/stalled",
                metric=step_key,
                message=f"did not increase (prev={prev_fmt}, current={curr_fmt})",
                tip="Verify rollouts are collected each epoch and dataloaders consume new batches.",
            )

    def _monitor_config_bounds(self, history: dict):
        """Check hard bounds (mathematical/physical limits) defined in metrics.yaml.

        These bounds represent impossible values that should never occur under correct
        implementation (e.g., negative probabilities, fractions > 1.0). Violations
        indicate bugs in the code, not just suboptimal training.
        """
        alerts: List[MetricAlert] = []
        for metric in history.keys():
            bounds = metrics_config.bounds_for_metric(metric)
            if not bounds:
                continue

            min_bound = bounds.get("min")
            max_bound = bounds.get("max")

            avg_value = self._windowed_metric_mean(
                history,
                metric,
                window=self._BOUNDS_SMOOTHING_WINDOW,
            )
            if avg_value is None:
                continue

            avg_with_latest = self._format_smoothed_with_latest(history, metric, avg_value)
            window_label = f"{self._BOUNDS_SMOOTHING_WINDOW}-step avg"

            if min_bound is not None and avg_value < float(min_bound):
                min_fmt = self._format_metric_value(metric, float(min_bound))
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/below_min",
                        metric=metric,
                        message=f"{window_label} {avg_with_latest} < hard minimum {min_fmt}",
                        tip="CRITICAL: This violates a mathematical invariant. Check implementation for bugs.",
                    )
                )

            if max_bound is not None and avg_value > float(max_bound):
                max_fmt = self._format_metric_value(metric, float(max_bound))
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/above_max",
                        metric=metric,
                        message=f"{window_label} {avg_with_latest} > hard maximum {max_fmt}",
                        tip="CRITICAL: This violates a mathematical invariant. Check implementation for bugs.",
                    )
                )

        return alerts or None

    def _monitor_entropy_collapse(self, history: dict):
        """Detect rapid/early collapse in policy entropy across algorithms.

        Compares the recent smoothed entropy to an early-training baseline
        (first N steps where N is the smoothing window from PPOAlerts/Core).
        Triggers when the recent average falls below a small fraction of the
        early average, signaling premature determinism and loss of exploration.
        """
        metric_key = "train/opt/policy/entropy"
        series = self._metric_series(history, metric_key)

        # Use the same smoothing window as other bounds checks
        window = self._BOUNDS_SMOOTHING_WINDOW

        # Require enough history to form early and recent windows
        if not series or len(series) < (2 * window):
            return None

        early_vals = [float(v) for _, v in series[:window]]
        if not early_vals:
            return None

        early_avg = fmean(early_vals)
        if early_avg <= 0.0:
            return None

        current_avg = self._windowed_metric_mean(history, metric_key, window=window)
        if current_avg is None:
            return None

        current = float(current_avg)

        # Trigger if recent entropy is a small fraction of the early baseline
        ratio = current / early_avg if early_avg else 1.0
        collapse_ratio_threshold = 0.25  # recent <= 25% of early baseline

        if ratio < collapse_ratio_threshold:
            early_fmt = self._format_metric_value(metric_key, early_avg)
            current_with_latest = self._format_smoothed_with_latest(history, metric_key, current)

            return MetricAlert(
                _id=f"{metric_key}/collapse",
                metric=metric_key,
                message=f"{window}-step avg {current_with_latest} vs early {early_fmt} (ratio {ratio:.2f})",
                tip="Increase entropy bonus (hp/ent_coef) or reduce over-updating (epochs/lr).",
            )

        return None

    # ---- initial policy sanity checks ----
    def _get_discrete_action_space(self):
        """Return discrete action space or None."""
        if self.pl_module is None:
            return None
        env = self.pl_module.get_env("train")
        action_space = getattr(env, "action_space", None)
        is_discrete = (
            action_space is not None
            and gym_spaces is not None
            and isinstance(action_space, gym_spaces.Discrete)
        )
        return action_space if is_discrete else None

    def _check_initial_action_uniform(
        self,
        history: dict,
        metric_key: str,
        expected_fn: Callable[[int], float],
        tolerance_fn: Callable[[float, int], float],
        alert_id_suffix: str,
        tip_message: str,
    ):
        """Generic helper for initial action uniformity checks."""
        action_space = self._get_discrete_action_space()
        if action_space is None:
            return None

        n = int(action_space.n)
        if n <= 1:
            return None

        expected = expected_fn(n)

        series = self._metric_series(history, metric_key)
        if not series:
            return None

        window = self._BOUNDS_SMOOTHING_WINDOW
        early_vals = [float(raw) for _, raw in series[:window]]
        if not early_vals:
            return None
        early_avg = fmean(early_vals)

        tol = tolerance_fn(expected, n)
        if math.fabs(early_avg - expected) <= tol:
            return None

        early_with_latest = self._format_smoothed_with_latest(history, metric_key, early_avg)
        exp_fmt = self._format_metric_value(metric_key, expected)

        return MetricAlert(
            _id=f"{metric_key}/{alert_id_suffix}",
            metric=metric_key,
            message=f"{window}-step early avg {early_with_latest} vs expected {exp_fmt} (n={n})",
            tip=tip_message,
        )

    def _monitor_initial_action_mean_uniform(self, history: dict):
        """Warn if early action mean deviates from uniform-policy expectation."""
        return self._check_initial_action_uniform(
            history,
            metric_key="train/roll/actions/mean",
            expected_fn=lambda n: (n - 1) / 2.0,
            tolerance_fn=lambda expected, n: 0.15 * float(max(1, n - 1)),
            alert_id_suffix="initial_uniform_mean_oob",
            tip_message="Initial policy may be biased; verify weight init and zero biases for policy head.",
        )

    def _monitor_initial_action_std_uniform(self, history: dict):
        """Warn if early action std deviates from uniform-policy expectation."""
        return self._check_initial_action_uniform(
            history,
            metric_key="train/roll/actions/std",
            expected_fn=lambda n: math.sqrt((n * n - 1) / 12.0),
            tolerance_fn=lambda expected, n: expected * 0.20,
            alert_id_suffix="initial_uniform_std_oob",
            tip_message="Initial action variability differs from uniform; check logits init and sampling path.",
        )

    def _check_episode_reward_trend(
        self,
        history: dict,
        alert_id_suffix: str,
        condition_fn: Callable[[float, float, float], bool],
        message_template: str,
        tip_message: str,
        threshold_ratio: float,
        threshold_min: float,
    ):
        """Generic helper for episode reward trend monitoring."""
        alerts: List[MetricAlert] = []
        window = self._EP_REWARD_WINDOW
        min_points = window * 2

        for metric_key in self._episode_reward_metric_keys(history):
            series = self._metric_series(history, metric_key)
            if len(series) < min_points:
                continue

            values = self._numeric_series(series)
            if len(values) < min_points:
                continue

            recent = values[-window:]
            prior = values[-(2 * window):-window]
            if len(recent) < window or len(prior) < window:
                continue

            recent_mean = fmean(recent)
            prior_mean = fmean(prior)
            delta = recent_mean - prior_mean
            threshold = max(
                math.fabs(prior_mean) * threshold_ratio,
                math.fabs(recent_mean) * threshold_ratio,
                threshold_min,
            )

            if not condition_fn(delta, threshold, prior_mean - recent_mean):
                continue

            precision = metrics_config.precision_for_metric(metric_key)
            recent_fmt = self._format_metric_value(metric_key, recent_mean)
            prior_fmt = self._format_metric_value(metric_key, prior_mean)
            delta_fmt = f"{delta:+.{precision}f}"
            threshold_fmt = f"{threshold:.{precision}f}"

            alerts.append(
                MetricAlert(
                    _id=f"{metric_key}/{alert_id_suffix}",
                    metric=metric_key,
                    message=message_template.format(
                        window=window,
                        recent_fmt=recent_fmt,
                        prior_fmt=prior_fmt,
                        delta_fmt=delta_fmt,
                        threshold_fmt=threshold_fmt,
                    ),
                    tip=tip_message,
                )
            )

        return alerts or None

    def _monitor_episode_reward_stalling(self, history: dict):
        return self._check_episode_reward_trend(
            history,
            alert_id_suffix="stalling",
            condition_fn=lambda delta, threshold, drop: math.fabs(delta) <= threshold,
            message_template="{window}-step mean {recent_fmt} vs prior {prior_fmt} (Δ={delta_fmt}, tol≤{threshold_fmt})",
            tip_message="Episode rewards plateaued; tweak lr, entropy bonus, or curriculum to regain momentum.",
            threshold_ratio=self._EP_REWARD_STALL_DELTA_RATIO,
            threshold_min=self._EP_REWARD_STALL_DELTA_MIN,
        )

    def _monitor_episode_reward_downward_trend(self, history: dict):
        return self._check_episode_reward_trend(
            history,
            alert_id_suffix="downward_trend",
            condition_fn=lambda delta, threshold, drop: drop >= threshold,
            message_template="{window}-step mean {recent_fmt} dropped from {prior_fmt} (Δ={delta_fmt}, threshold={threshold_fmt})",
            tip_message="Episode rewards regressed; lower lr, revisit exploration, or inspect reward scaling for regressions.",
            threshold_ratio=self._EP_REWARD_DECLINE_RATIO,
            threshold_min=self._EP_REWARD_DECLINE_MIN,
        )
