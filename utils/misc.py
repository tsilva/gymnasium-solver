"""Assorted small helpers (keep this file minimal).

Historically this module collected many unrelated utilities. These have been split into:
- utils.random_utils: randomness helpers (get_global_torch_generator)
- utils.torch_utils: torch helpers (inference_ctx, _device_of)
- utils.table_printer: NamespaceTablePrinter and print_namespaced_dict
- utils.dict_utils: prefix_dict_keys
- utils.stats_utils: calculate_deque_stats

Only truly miscellaneous helpers should live here going forward.
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np


def calculate_deque_stats(values_deque: deque, return_distribution: bool = False) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Calculate mean and standard deviation from a deque of values.

    Args:
        values_deque: Deque containing numeric values
        return_distribution: If True, also return the full array for distribution analysis

    Returns:
        Tuple of (mean, std, distribution_array) where distribution_array is None
        if return_distribution is False or deque is empty
    """
    if not values_deque:
        return 0.0, 0.0, None

    values_array = np.array(list(values_deque))
    mean = float(np.mean(values_array))
    std = float(np.std(values_array))
    distribution = values_array if return_distribution else None

    return mean, std, distribution