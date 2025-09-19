from __future__ import annotations

import csv
from pathlib import Path

from loggers.metrics_csv_logger import MetricsCSVLogger


def test_metrics_csv_respects_precision_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    logger = MetricsCSVLogger(csv_path)

    # Mix of counters (precision 0) and float metrics
    metrics = {
        "train/total_timesteps": 100.0,  # precision 0 -> int
        "train/epoch": 5.0,               # precision 0 -> int
        "train/ep_len_mean": 12.9,       # precision 0 -> int (rounded)
        "train/entropy": 0.123456,       # precision 4 -> rounded float
    }

    logger.buffer_metrics(metrics)
    logger.close()

    # Read back and validate string representations in CSV
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]

    # Base fields coerced
    assert row["total_timesteps"] == "100"
    assert row["epoch"] == "5"

    # Namespaced metric columns
    assert row["train/ep_len_mean"] == "13"  # 12.9 -> 13 (precision 0)
    assert row["train/entropy"] == "0.1235"  # rounded to 4 decimals

