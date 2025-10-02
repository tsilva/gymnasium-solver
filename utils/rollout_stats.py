from collections import deque

import numpy as np


class RollingWindow:
    """O(1) rolling window with deque semantics and constant-time mean()."""

    def __init__(self, maxlen: int):
        if maxlen <= 0: raise ValueError("RollingWindow maxlen must be > 0")
        self._maxlen = int(maxlen)
        self._dq = deque()
        self._sum = 0.0

    def append(self, value: float) -> None:
        if len(self._dq) == self._maxlen:
            oldest = self._dq.popleft()
            self._sum -= float(oldest)
        self._dq.append(value)
        self._sum += float(value)

    def mean(self) -> float:
        n = len(self._dq)
        if n == 0: return 0.0
        return self._sum / n

    def __len__(self) -> int:
        return len(self._dq)

    def __bool__(self) -> bool:
        return len(self._dq) > 0


class RunningStats:
    """Constant-time mean/std aggregates for streaming numeric updates."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_squared = 0.0

    def update(self, values: np.ndarray) -> None:
        # Fast path: skip stats collection if values are empty
        if values.size == 0: return

        self.count += values.size

        # Work with flattened view directly (no copy)
        vals_flat = values.ravel()

        # Convert to float32 only if needed
        if vals_flat.dtype != np.float32:
            vals_flat = vals_flat.astype(np.float32)

        # Accumulate stats (keep in float32 for speed)
        self.sum += float(vals_flat.sum())
        self.sum_squared += float((vals_flat * vals_flat).sum())

    def mean(self) -> float:
        if self.count == 0: return 0.0
        return self.sum / self.count

    def std(self) -> float:
        if self.count == 0: return 0.0
        mean = self.mean()
        var = max(0.0, self.sum_squared / self.count - mean * mean)
        return float(np.sqrt(var))
