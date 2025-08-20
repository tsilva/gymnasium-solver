from __future__ import annotations
from typing import Callable, Dict, Mapping, MutableMapping
from .dict_utils import prefix_dict_keys

class MetricsBuffer:
    """
    A simple buffer to collect metrics during training or evaluation.
    This is useful for aggregating metrics before logging them to a logger
    or for further processing. 
    
    NOTE: This bypasses the a training bottleneck issue with using the Lightning's 
    logging facilities multiple times per step/epoch; this way we can log metrics
    multiple times per step/epoch and only flush them to the logger at the end of the epoch.
    """
    def __init__(self) -> None:
        self._data: Dict[str, list] = {}

    def log(self, metrics: Mapping[str, float], prefix: str | None = None) -> None:
        _metrics = metrics if prefix is None else prefix_dict_keys(metrics, prefix)
        for key, value in _metrics.items():
            self._data.setdefault(key, []).append(value)

    def means(self) -> Dict[str, float]:
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self._data.items()}

    def clear(self) -> None:
        self._data.clear()

    def flush_to(self, log_fn: Callable[[MutableMapping[str, float]], None]) -> Dict[str, float]:
        """
        Flush means to the provided logger function (e.g., LightningModule.log_dict)
        and clear internal buffers. Returns the computed means for external sinks.
        """

        # Calculate metric means (eg: epoch averages)
        means = self.means()

        # Logs the means to the provided logger function (eg: LightningModule.log_dict)
        log_fn(means)

        # Clear the buffer after logging (eg: fresh data for the next epoch)
        self.clear()

        # Return the means for any further processing
        return means
