from __future__ import annotations

from typing import Iterable, List

from agents.ppo.ppo_alerts import PPOAlerts
from utils.metric_bundles import CommonMetricAlerts
from utils.metrics_config import metrics_config
from utils.metrics_monitor import MetricsMonitor
from utils.metrics_recorder import MetricsRecorder


def _collect_alerts(bundle, history: dict) -> List:
    alerts = []
    for fn in bundle.get_monitor_fns():
        result = fn(history)
        if not result:
            continue
        if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
            alerts.extend(result)
        else:
            alerts.append(result)
    return alerts


def test_common_alerts_flag_nan_metric():
    recorder = MetricsRecorder()
    step_key = metrics_config.step_key()
    recorder.update_history({
        step_key: 128,
        "train/loss": float("nan"),
    })
    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CommonMetricAlerts())

    result = monitor.check()

    assert "train/loss" in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts["train/loss"]}
    assert "train/loss/nan_or_inf" in alert_ids


def test_common_alerts_detect_step_stall():
    recorder = MetricsRecorder()
    step_key = metrics_config.step_key()
    recorder.update_history({step_key: 256})
    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CommonMetricAlerts())

    # Simulate another epoch without additional progress
    recorder.update_history({step_key: 256})
    result = monitor.check()

    assert step_key in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts[step_key]}
    assert f"{step_key}/stalled" in alert_ids


def test_common_alerts_guard_config_bounds():
    recorder = MetricsRecorder()
    step_key = metrics_config.step_key()
    recorder.update_history({
        step_key: 512,
        "train/clip_fraction": 1.5,
    })
    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CommonMetricAlerts())

    result = monitor.check()

    assert "train/clip_fraction" in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts["train/clip_fraction"]}
    assert "train/clip_fraction/above_max" in alert_ids


def test_ppo_alerts_explained_variance_low():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/explained_variance": [(1000, -0.4)],
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/explained_variance/too_low" in alert_ids


def test_ppo_alerts_explained_variance_high():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/explained_variance": [(2000, 1.2)],
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/explained_variance/too_high" in alert_ids


def test_ppo_alerts_clip_fraction_spike_not_flagged_by_average():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/clip_fraction": [
            (1, 0.02),
            (2, 0.02),
            (3, 0.02),
            (4, 0.8),
        ]
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/clip_fraction/oob_max" not in alert_ids


def test_ppo_alerts_clip_fraction_high_average_triggers():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/clip_fraction": [
            (1, 0.62),
            (2, 0.58),
            (3, 0.61),
            (4, 0.64),
        ]
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/clip_fraction/oob_max" in alert_ids
