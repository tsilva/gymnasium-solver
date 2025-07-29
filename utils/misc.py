import torch
import itertools
from typing import Dict, Any
from contextlib import contextmanager


def prefix_dict_keys(data: dict, prefix: str) -> dict:
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}

def print_namespaced_dict(data: dict) -> None:
    """
    Prints a dictionary with namespaced keys (e.g., 'rollout/ep_len_mean')
    in a formatted ASCII table grouped by namespaces.
    Floats are formatted to 2 decimal places.
    """
    if not data: return
    # Group keys by their namespace prefix
    grouped = {}
    for key, value in data.items():
        if "/" in key:
            namespace, subkey = key.split("/", 1)
        else:
            namespace, subkey = key, ""
        grouped.setdefault(namespace, {})[subkey] = value

    # Format values first (floats to 2 decimals)
    formatted_grouped = {}
    for ns, subdict in grouped.items():
        formatted_grouped[ns] = {
            subkey: str(val)
            for subkey, val in subdict.items()
        }

    # Determine column widths
    max_key_len = max(len(subkey) for ns in formatted_grouped for subkey in formatted_grouped[ns]) + 4
    max_val_len = max(len(val) for ns in formatted_grouped for val in formatted_grouped[ns].values()) + 2

    # Print table
    border = "-" * (max_key_len + max_val_len + 5)
    print(border)
    for ns, subdict in formatted_grouped.items():
        print(f"| {ns + '/':<{max_key_len}} |")
        for subkey, val in subdict.items():
            print(f"|    {subkey:<{max_key_len-4}} | {val:<{max_val_len}}|")
    print(border)

def _convert_numeric_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string representations of numbers back to numeric types."""
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            if 'e' in value.lower() or 'E' in value:
                try:
                    config_dict[key] = float(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
    return config_dict


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
