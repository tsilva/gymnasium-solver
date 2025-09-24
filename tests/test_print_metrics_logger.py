from __future__ import annotations

import io

from loggers.metrics_table_logger import MetricsTableLogger
from utils.logging import strip_ansi_codes
from utils.metrics_monitor import MetricsMonitor
from utils.metrics_recorder import MetricsRecorder


def _make_logger():
    buf = io.StringIO()
    recorder = MetricsRecorder(step_key="train/total_timesteps")
    monitor = MetricsMonitor(recorder)
    logger = MetricsTableLogger(metrics_monitor=monitor)
    logger.use_ansi_inplace = True
    logger.stream = buf
    return logger, buf


def test_print_logger_drops_bare_epoch_when_namespaced_exists():
    # Create logger with a StringIO stream to capture output
    logger, buf = _make_logger()

    # Provide metrics containing both bare 'epoch' and namespaced 'train/epoch'
    metrics = {
        "epoch": 3,
        "train/epoch": 3,
        "train/total_timesteps": 128.0,
        "train/ep_rew/mean": 10.0,
    }

    logger.log_metrics(metrics)

    out = buf.getvalue()

    # Should not create a separate 'epoch/' section header
    assert "epoch/" not in out

    # Sanity: train section should be present
    assert "train/" in out


def test_print_logger_preserves_sticky_values_between_calls():
    # Create logger with in-place updates to avoid clears
    logger, buf = _make_logger()

    # First payload includes ep_rew/mean
    first = {
        "train/total_timesteps": 128.0,
        "train/ep_rew/mean": 10.0,
    }
    logger.log_metrics(first)

    # Reset buffer to capture only the second render
    buf.truncate(0)
    buf.seek(0)

    # Second payload updates only total_timesteps; ep_rew/mean omitted
    second = {
        "train/total_timesteps": 256.0,
    }
    logger.log_metrics(second)

    out = strip_ansi_codes(buf.getvalue())

    # Sticky behavior: ep_rew/mean should still be present with last value
    assert "ep_rew/mean" in out
    assert "10.00" in out or "10" in out


def test_print_logger_trigger_alert_highlight_persists_and_clears():
    logger, buf = _make_logger()
    logger.colors_enabled = False  # simulate non-TTY; alert highlighting should still force color
    logger.metrics_monitor.active_alerts = {
        "train/ep_rew/mean": ["training reward below threshold"],
    }

    # First payload includes an alert, which should activate yellow row highlight
    first = {
        "train/ep_rew/mean": 5.0,
    }
    logger.log_metrics(first)

    out_with_alert = buf.getvalue()
    assert "\x1b[43m" in out_with_alert  # bg_yellow applied to the row
    assert "training reward below threshold" in out_with_alert
    assert "train/ep_rew/mean" in logger.metrics_monitor.active_alerts

    # Second payload omits the alert key, which should clear highlight state
    buf.truncate(0)
    buf.seek(0)
    logger.metrics_monitor.active_alerts = {}
    second = {
        "train/ep_rew/mean": 6.0,
    }
    logger.log_metrics(second)

    out_after_clear = buf.getvalue()
    assert "\x1b[43m" not in out_after_clear
    assert "train/ep_rew/mean" not in logger.metrics_monitor.active_alerts
