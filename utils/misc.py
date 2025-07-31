import torch
import itertools
from typing import Dict, Any
from contextlib import contextmanager
import os
import sys
from typing import Dict, Any


def prefix_dict_keys(data: dict, prefix: str) -> dict:
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}


# TODO: move this somewhere else?
@contextmanager
def inference_ctx(*modules):
    """
    Temporarily puts all passed nn.Module objects in eval mode and
    disables grad-tracking. Restores their original training flag
    afterwards.

    Usage:
        with inference_ctx(actor, critic):
            ... collect trajectories ...
    """
    # Filter out Nones and flatten (in case you pass lists/tuples)
    flat = [m for m in itertools.chain.from_iterable(
            (m if isinstance(m, (list, tuple)) else (m,)) for m in modules)
            if m is not None]

    # Remember original .training flags
    was_training = [m.training for m in flat]
    try:
        for m in flat:
            m.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, flag in zip(flat, was_training):
            if flag:   # only restore if it *was* in train mode
                m.train()

# TODO: move this somewhere else?
def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device



def print_namespaced_dict(
    data: Dict[str, Any],
    inplace: bool = True,
    float_fmt: str = ".2f",
    indent: int = 4,
) -> None:
    """
    Prints a dictionary with namespaced keys (e.g., 'rollout/ep_len_mean')
    as a formatted ASCII table grouped by namespaces.

    - Namespace headers (e.g., 'train/', 'time/', 'rollout/') are flush-left.
    - Metric rows are indented by `indent` spaces.
    - Floats are formatted using `float_fmt` (default: '.2f').
    """
    if not data:
        return

    if inplace:
        os.system('cls' if os.name == 'nt' else 'clear')

    # Group by namespace (preserves insertion order in Python 3.7+)
    grouped: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        ns, sub = k.split("/", 1) if "/" in k else (k, "")
        grouped.setdefault(ns, {})[sub] = v

    # Formatter
    def fmt(v: Any) -> str:
        return format(v, float_fmt) if isinstance(v, float) else str(v)

    # Compute column widths (include headers and all subkeys/values)
    ns_order = list(grouped.keys())
    key_candidates = [ns + "/" for ns in ns_order]  # headers
    val_candidates = []
    for ns in ns_order:
        for sub, v in grouped[ns].items():
            key_candidates.append(sub)
            val_candidates.append(fmt(v))

    key_width = max(len(k) for k in key_candidates) if key_candidates else 0
    val_width = max((len(v) for v in val_candidates), default=0)

    # Total width matches the row layout below:
    # "| " + (key field of width indent+key_width) + " | " + (val field of width val_width) + " |"
    border_len = 2 + (indent + key_width) + 3 + val_width + 2
    border = "-" * border_len

    print(border)
    for ns in ns_order:
        header = ns + "/"
        # Header: NO indent visually; we fill the whole key field without the leading indent spaces
        # but keep the field width equal to (indent + key_width) so the right column aligns.
        print(f"| {header:<{indent + key_width}} | {'':>{val_width}} |")

        # Metrics: keep indent before subkeys
        for sub, v in grouped[ns].items():
            sub_disp = sub  # empty subkey allowed
            val_disp = fmt(v)
            print(f"| {' ' * indent}{sub_disp:<{key_width}} | {val_disp:>{val_width}} |")
    print(border)