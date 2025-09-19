from statistics import fmean
from typing import Callable, Iterable, List

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

        values = []
        for _, raw in series[-window:]:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue

        if not values:
            return None

        return fmean(values)

    def _format_metric_value(self, metric_key: str, value: float) -> str:
        precision = metrics_config.precision_for_metric(metric_key)
        format_str = f"{{:.{precision}f}}"
        return format_str.format(float(value))


# TODO: call core
class CoreMetricAlerts(MetricMonitorBundle):
    """Shared metric tripwires that apply to all algorithms."""

    _BOUNDS_SMOOTHING_WINDOW = 5

    def __init__(self, pl_module: pl.LightningModule) -> None:
        self.pl_module = pl_module
        self._step_key = metrics_config.total_timesteps_key()

    # ---- monitors ----
    def _monitor_step_progress(self, history: dict):
        if self.pl_module is None:
            return None
        step = self.pl_module.current_epoch
        if step is None:
            return None
        step_key = f"{self._step_key}/{step}"
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

    # TODO: tag for removal
    def _monitor_config_bounds(self, history: dict):
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

            latest_value = self._latest_metric_value(history, metric)
            latest_fmt: str | None = None
            if latest_value is not None:
                try:
                    latest_fmt = self._format_metric_value(metric, float(latest_value))
                except (TypeError, ValueError):
                    latest_fmt = None

            avg_fmt = self._format_metric_value(metric, avg_value)
            window_label = f"{self._BOUNDS_SMOOTHING_WINDOW}-step avg"
            latest_suffix = f" (latest {latest_fmt})" if latest_fmt else ""

            if min_bound is not None and avg_value < float(min_bound):
                min_fmt = self._format_metric_value(metric, float(min_bound))
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/below_min",
                        metric=metric,
                        message=f"{window_label} {avg_fmt} < configured min {min_fmt}{latest_suffix}",
                        tip="Reduce aggressiveness (e.g., learning rate, clipping) or review reward scaling.",
                    )
                )

            if max_bound is not None and avg_value > float(max_bound):
                max_fmt = self._format_metric_value(metric, float(max_bound))
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/above_max",
                        metric=metric,
                        message=f"{window_label} {avg_fmt} > configured max {max_fmt}{latest_suffix}",
                        tip="Adjust hyperparameters or enable stronger regularization to pull the metric back in range.",
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
        metric_key = "train/entropy"
        series = self._metric_series(history, metric_key)

        # Use the same smoothing window as other bounds checks
        window = self._BOUNDS_SMOOTHING_WINDOW

        # Require enough history to form early and recent windows
        if not series or len(series) < (2 * window):
            return None

        try:
            early_vals = [float(v) for _, v in series[:window]]
        except (TypeError, ValueError):
            early_vals = []
        if not early_vals:
            return None

        early_avg = fmean(early_vals)
        if early_avg <= 0.0:
            return None

        current_avg = self._windowed_metric_mean(history, metric_key, window=window)
        if current_avg is None:
            return None

        try:
            current = float(current_avg)
        except (TypeError, ValueError):
            return None

        # Trigger if recent entropy is a small fraction of the early baseline
        ratio = current / early_avg if early_avg else 1.0
        collapse_ratio_threshold = 0.25  # recent <= 25% of early baseline

        if ratio < collapse_ratio_threshold:
            early_fmt = self._format_metric_value(metric_key, early_avg)
            # Include both smoothed and latest values for context
            smoothed_fmt = self._format_metric_value(metric_key, current)
            latest_val = self._latest_metric_value(history, metric_key)
            latest_suffix = ""
            try:
                if latest_val is not None:
                    latest_fmt = self._format_metric_value(metric_key, float(latest_val))
                    latest_suffix = f" (latest {latest_fmt})"
            except (TypeError, ValueError):
                latest_suffix = ""

            return MetricAlert(
                _id=f"{metric_key}/collapse",
                metric=metric_key,
                message=f"{window}-step avg {smoothed_fmt}{latest_suffix} vs early {early_fmt} (ratio {ratio:.2f})",
                tip="Increase entropy bonus (hp/ent_coef) or reduce over-updating (epochs/lr).",
            )

        return None
