from typing import Callable, Iterable, List

import math
from statistics import fmean

import pytorch_lightning as pl

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
