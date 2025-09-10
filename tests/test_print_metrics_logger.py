from __future__ import annotations

import io

from loggers.print_metrics_logger import PrintMetricsLogger


def test_print_logger_drops_bare_epoch_when_namespaced_exists():
    # Create logger with a StringIO stream to capture output
    logger = PrintMetricsLogger()
    logger.use_ansi_inplace = True  # avoid clearing the terminal
    buf = io.StringIO()
    logger.stream = buf

    # Provide metrics containing both bare 'epoch' and namespaced 'train/epoch'
    metrics = {
        "epoch": 3,
        "train/epoch": 3,
        "train/total_timesteps": 128.0,
        "train/ep_rew_mean": 10.0,
    }

    logger.log_metrics(metrics)

    out = buf.getvalue()

    # Should not create a separate 'epoch/' section header
    assert "epoch/" not in out

    # Sanity: train section should be present
    assert "train/" in out

