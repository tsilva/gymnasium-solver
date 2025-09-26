from __future__ import annotations

from typing import Iterable, List

from agents.ppo.ppo_alerts import PPOAlerts
from utils.metric_bundles import CoreMetricAlerts
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
    step_key = metrics_config.total_timesteps_key()
    recorder.update_history({
        step_key: 128,
        "train/loss/total": float("nan"),
    })
    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CoreMetricAlerts())

    result = monitor.check()

    assert "train/loss/total" in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts["train/loss/total"]}
    assert "train/loss/total/nan_or_inf" in alert_ids


def test_common_alerts_detect_step_stall():
    recorder = MetricsRecorder()
    step_key = metrics_config.total_timesteps_key()
    recorder.update_history({step_key: 256})
    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CoreMetricAlerts())

    # Simulate another epoch without additional progress
    recorder.update_history({step_key: 256})
    result = monitor.check()

    assert step_key in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts[step_key]}
    assert f"{step_key}/stalled" in alert_ids


def test_common_alerts_guard_config_bounds():
    recorder = MetricsRecorder()
    step_key = metrics_config.total_timesteps_key()
    recorder.update_history({
        step_key: 512,
        "train/opt/ppo/clip_fraction": 1.5,
    })
    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CoreMetricAlerts())

    result = monitor.check()

    assert "train/opt/ppo/clip_fraction" in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts["train/opt/ppo/clip_fraction"]}
    assert "train/opt/ppo/clip_fraction/above_max" in alert_ids


def test_common_alerts_detect_episode_reward_stall():
    recorder = MetricsRecorder()
    step_key = metrics_config.total_timesteps_key()
    reward_key = "train/roll/ep_rew/mean"

    values = [
        0.0,
        1.0,
        2.0,
        4.0,
        9.9,
        10.1,
        10.0,
        9.95,
        10.05,
        10.02,
        10.01,
        9.98,
        10.04,
        10.00,
        10.02,
        9.99,
        10.01,
        10.00,
        9.97,
        10.03,
    ]

    for idx, value in enumerate(values, start=1):
        recorder.update_history({
            step_key: idx * 128,
            reward_key: value,
        })

    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CoreMetricAlerts())
    result = monitor.check()

    assert reward_key in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts[reward_key]}
    assert f"{reward_key}/stalling" in alert_ids


def test_common_alerts_detect_episode_reward_downward_trend():
    recorder = MetricsRecorder()
    step_key = metrics_config.total_timesteps_key()
    reward_key = "train/roll/ep_rew/mean"

    values = [
        0.0,
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
        11.0,
        13.0,
        20.0,
        21.0,
        19.5,
        20.4,
        20.2,
        19.9,
        20.1,
        20.3,
        15.0,
        14.8,
        14.5,
        14.2,
        14.0,
        13.8,
        13.5,
        13.2,
    ]

    for idx, value in enumerate(values, start=1):
        recorder.update_history({
            step_key: idx * 256,
            reward_key: value,
        })

    monitor = MetricsMonitor(recorder)
    monitor.register_bundle(CoreMetricAlerts())
    result = monitor.check()

    assert reward_key in result["added"]
    alert_ids = {alert._id for alert in monitor.active_alerts[reward_key]}
    assert f"{reward_key}/downward_trend" in alert_ids


def test_ppo_alerts_explained_var_low():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/opt/value/explained_var": [(1000, -0.4)],
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/opt/value/explained_var/too_low" in alert_ids


def test_ppo_alerts_explained_var_high():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/opt/value/explained_var": [(2000, 1.2)],
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/opt/value/explained_var/too_high" in alert_ids


def test_ppo_alerts_clip_fraction_spike_not_flagged_by_average():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/opt/ppo/clip_fraction": [
            (1, 0.02),
            (2, 0.02),
            (3, 0.02),
            (4, 0.8),
        ]
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/opt/ppo/clip_fraction/oob_max" not in alert_ids


def test_ppo_alerts_clip_fraction_high_average_triggers():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/opt/ppo/clip_fraction": [
            (1, 0.62),
            (2, 0.58),
            (3, 0.61),
            (4, 0.64),
        ]
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/opt/ppo/clip_fraction/oob_max" in alert_ids


def test_ppo_alerts_explained_var_worse_than_mean():
    bundle = PPOAlerts(agent=None)
    history = {
        "train/opt/value/explained_var": [
            (10, -0.05),
            (11, -0.02),
            (12, -0.03),
            (13, -0.04),
            (14, -0.01),
        ]
    }

    alerts = _collect_alerts(bundle, history)
    alert_ids = {alert._id for alert in alerts}
    assert "train/opt/value/explained_var/worse_than_mean" in alert_ids
