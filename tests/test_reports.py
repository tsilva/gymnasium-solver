from io import StringIO
from contextlib import redirect_stdout

from utils.metrics_monitor import MetricAlert
from utils.reports import print_terminal_ascii_alerts


def test_print_terminal_ascii_alerts_includes_epoch_summary():
    alert = MetricAlert(
        _id="train/ep_rew_mean/too_low",
        metric="train/ep_rew_mean",
        message="training reward below threshold",
        tip="increase policy entropy",
    )

    freq_alerts = [
        {
            "alert": alert,
            "count": 3,
            "epoch_count": 3,
            "epochs": (0, 2, 5),
        }
    ]

    buf = StringIO()
    with redirect_stdout(buf):
        print_terminal_ascii_alerts(freq_alerts, total_epochs=10, width=24)

    output = buf.getvalue()

    assert "3/10 (30%)" in output
    assert "triggered in `3/10 (30%)` epochs of training" in output
