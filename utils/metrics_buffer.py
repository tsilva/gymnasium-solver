from __future__ import annotations

from typing import Callable, Dict, Mapping, MutableMapping

from .misc import prefix_dict_keys


class MetricsBuffer:
    """
    Lightweight buffer to collect metrics during an epoch and flush means once.

    Encapsulates BaseAgent's ad-hoc dict of lists and mean computation.
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

    def flush_to(self, log_fn: Callable[[MutableMapping[str, float]], None]) -> None:
        """
        Flush means to the provided logger function (e.g., LightningModule.log_dict)
        and clear internal buffers.
        """
        log_fn(self.means())
        self.clear()
