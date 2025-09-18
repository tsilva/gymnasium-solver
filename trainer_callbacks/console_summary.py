"""Console end-of-training summary callback.

Prints an ASCII sparkline summary of numeric metrics and a compact alerts
recap at the end of training. Presentation concerns are handled here rather
than inside the agent.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import pytorch_lightning as pl

from utils.reports import print_terminal_ascii_summary


class ConsoleSummaryCallback(pl.Callback):
    """Render a terminal summary at fit end using the agent's recorder/monitor."""

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Best-effort guard: only act when required attributes are present
        rec = getattr(pl_module, "metrics_recorder", None)
        mon = getattr(pl_module, "metrics_monitor", None)
        if rec is None:
            return

        # Print metrics history summary (sparklines + stats)
        try:
            history = rec.history()
            print_terminal_ascii_summary(history)
        except Exception:
            # Do not fail training teardown due to presentation issues
            pass

        # Print a compact alerts recap, sorted by frequency
        try:
            if mon is None:
                return
            counter = mon.get_alerts_counter()
            # Support either {metric: [message, count]} or {metric: count}
            def _count(v: Any) -> int:
                if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
                    try:
                        # typical shape: [message, count]
                        return int(list(v)[-1])
                    except Exception:
                        return 0
                try:
                    return int(v)
                except Exception:
                    return 0

            sorted_items = sorted(counter.items(), key=lambda kv: _count(kv[1]), reverse=True)
            if sorted_items:
                print("### ALERTS RAISED DURING TRAINING ###")
                for metric, val in sorted_items:
                    print(f"{metric}: {_count(val)}")
        except Exception:
            pass

