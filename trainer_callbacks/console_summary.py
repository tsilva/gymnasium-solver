"""Console end-of-training summary callback.

Prints an ASCII sparkline summary of numeric metrics and a compact alerts
recap at the end of training. Presentation concerns are handled here rather
than inside the agent.
"""

import pytorch_lightning as pl

# TODO: move these to this file
from utils.reports import (
    print_terminal_ascii_alerts,
    print_terminal_ascii_summary,
    print_training_completion_status,
)


# TODO: call this callback something more appropriate
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

        # Print training completion status
        # Get final rewards from metrics
        train_reward = None
        val_reward = None
        if "train/roll/ep_rew/mean" in history:
            train_data = history["train/roll/ep_rew/mean"]
            if train_data:
                train_reward = train_data[-1][1]  # (step, value)

        if "val/roll/ep_rew/mean" in history:
            val_data = history["val/roll/ep_rew/mean"]
            if val_data:
                val_reward = val_data[-1][1]  # (step, value)

        # Get threshold from config
        reward_threshold = getattr(pl_module.config, "reward_threshold", None)
        if reward_threshold is None:
            # Try to get from env spec
            spec = getattr(pl_module.config, "spec", {})
            returns = spec.get("returns", {})
            reward_threshold = returns.get("threshold_solved")

        # Get early stop reason if available
        early_stop_reason = getattr(pl_module, "_early_stop_reason", None)

        print_training_completion_status(
            train_reward=train_reward,
            val_reward=val_reward,
            reward_threshold=reward_threshold,
            early_stop_reason=early_stop_reason,
        )
