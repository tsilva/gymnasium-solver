"""Console end-of-training summary callback.

Prints an ASCII sparkline summary of numeric metrics and a compact alerts
recap at the end of training. Presentation concerns are handled here rather
than inside the agent.
"""

import pytorch_lightning as pl

from utils.reports import print_terminal_ascii_summary, print_terminal_ascii_alerts


class ConsoleSummaryCallback(pl.Callback):
    """Render a terminal summary at fit end using the agent's recorder/monitor."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Skip summary if training was aborted before it began
        if getattr(pl_module, "_aborted_before_training", False):
            return

        # Resolve metrics recorder and monitor
        metrics_recorder = pl_module.metrics_recorder
        metrics_monitor = pl_module.metrics_monitor

        # Print metrics history summary
        history = metrics_recorder.history()
        print_terminal_ascii_summary(history)

        # Print alerts
        freq_alerts = metrics_monitor.get_alerts_by_frequency()
        total_epochs = metrics_monitor.get_total_epochs()
        if freq_alerts: print_terminal_ascii_alerts(freq_alerts, total_epochs=total_epochs)
