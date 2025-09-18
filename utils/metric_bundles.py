from typing import Callable, Iterable, List

import math

from utils.metrics_config import metrics_config
from utils.metrics_monitor import MetricAlert


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


class CommonMetricAlerts(MetricMonitorBundle):
    """Shared metric tripwires that apply to all algorithms."""

    _DEFAULT_NAN_METRICS: tuple[str, ...] = (
        "train/loss",
        "train/entropy",
        "train/entropy_loss",
        "train/ep_rew_mean",
    )

    def __init__(self, *, nan_metrics: Iterable[str] | None = None) -> None:
        self._step_key = metrics_config.step_key()
        merged: List[str] = list(self._DEFAULT_NAN_METRICS)
        if nan_metrics:
            for metric in nan_metrics:
                if metric not in merged:
                    merged.append(metric)
        self._nan_watchlist: List[str] = merged

    def extend_nan_watchlist(self, metrics: Iterable[str]) -> None:
        for metric in metrics:
            if metric not in self._nan_watchlist:
                self._nan_watchlist.append(metric)

    # ---- helpers ----
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

    # ---- monitors ----
    def _monitor_step_progress(self, history: dict):
        series = history.get(self._step_key)
        if not series or len(series) < 2:
            return None
        _, prev_value = series[-2]
        _, current_value = series[-1]
        if current_value <= prev_value:
            prev_fmt = self._format_metric_value(self._step_key, float(prev_value))
            curr_fmt = self._format_metric_value(self._step_key, float(current_value))
            return MetricAlert(
                _id=f"{self._step_key}/stalled",
                metric=self._step_key,
                message=f"did not increase (prev={prev_fmt}, current={curr_fmt})",
                tip="Verify rollouts are collected each epoch and dataloaders consume new batches.",
            )

    def _monitor_nan_metrics(self, history: dict):
        alerts: List[MetricAlert] = []
        for metric in self._nan_watchlist:
            value = self._latest_metric_value(history, metric)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric) or math.isinf(numeric):
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/nan_or_inf",
                        metric=metric,
                        message="recorded NaN/Inf; training likely diverged",
                        tip="Inspect gradients, learning rate, and normalization to regain numerical stability.",
                    )
                )
        return alerts or None

    def _monitor_config_bounds(self, history: dict):
        alerts: List[MetricAlert] = []
        for metric, series in history.items():
            if not series:
                continue
            _, raw_value = series[-1]
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            bounds = metrics_config.bounds_for_metric(metric)
            if not bounds:
                continue

            min_bound = bounds.get("min")
            max_bound = bounds.get("max")

            if min_bound is not None and value < float(min_bound):
                min_fmt = self._format_metric_value(metric, float(min_bound))
                val_fmt = self._format_metric_value(metric, value)
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/below_min",
                        metric=metric,
                        message=f"{val_fmt} < configured min {min_fmt}",
                        tip="Reduce aggressiveness (e.g., learning rate, clipping) or review reward scaling.",
                    )
                )

            if max_bound is not None and value > float(max_bound):
                max_fmt = self._format_metric_value(metric, float(max_bound))
                val_fmt = self._format_metric_value(metric, value)
                alerts.append(
                    MetricAlert(
                        _id=f"{metric}/above_max",
                        metric=metric,
                        message=f"{val_fmt} > configured max {max_fmt}",
                        tip="Adjust hyperparameters or enable stronger regularization to pull the metric back in range.",
                    )
                )

        return alerts or None
