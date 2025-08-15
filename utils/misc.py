import torch
import itertools
from typing import Dict, Any
from contextlib import contextmanager
import os
import sys
from typing import Dict, Any, Iterable, Optional, List, Tuple
import numbers
import torch
import numpy as np
from collections import deque

# ========= Randomness helpers =========

_global_torch_generator: Optional[torch.Generator] = None

def get_global_torch_generator(seed: Optional[int] = None) -> torch.Generator:
        """Return a process-wide torch.Generator, optionally seeded once.

        Notes:
        - The first call can pass a seed to initialize determinism. Subsequent calls
            will return the same generator instance and ignore the seed argument.
        - Using a single shared generator keeps shuffles reproducible across users
            without tightly coupling callsites to local seeding logic.
        """
        global _global_torch_generator
        if _global_torch_generator is None:
                g = torch.Generator()
                if seed is not None:
                        g.manual_seed(int(seed))
                _global_torch_generator = g
        return _global_torch_generator

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

def prefix_dict_keys(data: dict, prefix: str) -> dict:
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}

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
    # TODO: double check that this isn't causing issues
    import torch.nn as nn
    modules = [m for m in modules if isinstance(m, nn.Module)]
    
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
    if not hasattr(module, 'parameters'): return torch.device('cpu')
    return next(module.parameters()).device


# ========= Helpers =========

def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Number)

def _humanize_num(v: numbers.Number, float_fmt: str = ".2f") -> str:
    """Compact formatting for ints/floats."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        n = abs(v)
        sign = "-" if v < 0 else ""
        if n >= 1_000_000_000:
            return f"{sign}{n/1_000_000_000:.2f}B"
        if n >= 1_000_000:
            return f"{sign}{n/1_000_000:.2f}M"
        if n >= 1_000:
            return f"{sign}{n/1_000:.2f}k"
        return str(v)
    if isinstance(v, float):
        # keep small floats readable; switch to scientific for very tiny
        if 0 < abs(v) < 1e-6:
            return f"{v:.2e}"
        return format(v, float_fmt)
    return str(v)

def _fmt_plain(v: Any, float_fmt: str = ".2f") -> str:
    if isinstance(v, float):
        return format(v, float_fmt)
    return str(v)

def _ansi(color: Optional[str], s: str, enable: bool) -> str:
    if not enable or not color:
        return s
    codes = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
        "bold": "1",
    }
    if color == "bold":
        return f"\x1b[{codes['bold']}m{s}\x1b[0m"
    return f"\x1b[{codes[color]}m{s}\x1b[0m"

# ========= Printer =========

class NamespaceTablePrinter:
    def __init__(
        self,
        *,
        float_fmt: str = ".2f",
        indent: int = 4,
        compact_numbers: bool = True,
        color: bool = True,
        # If provided, use this to decide what counts as "improvement".
        # Keys are "namespace/subkey" (e.g. "train/loss": False meaning lower is better).
        better_when_increasing: Optional[Dict[str, bool]] = None,
        # Section order; others follow alphabetically
        fixed_section_order: Optional[Iterable[str]] = ("train", "eval"),
        sort_keys_within_section: bool = True,
        # Render method
        use_ansi_inplace: bool = True,
        stream=None,
        delta_tol: float = 1e-12,
        # Precision per metric: {"namespace/subkey": precision, ...} where 0 = int
        metric_precision: Optional[Dict[str, int]] = None,
        # Minimum width for the values column
        min_val_width: int = 15,
        # Priority order for sorting keys within sections
        key_priority: Optional[List[str]] = None,
    ):
        self.float_fmt = float_fmt
        self.indent = indent
        self.compact_numbers = compact_numbers
        self.color = bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        self.better_when_increasing = dict(better_when_increasing or {})
        self.fixed_section_order = list(fixed_section_order) if fixed_section_order else None
        self.sort_keys_within_section = sort_keys_within_section
        self.use_ansi_inplace = use_ansi_inplace and sys.stdout.isatty()
        self.stream = stream or sys.stdout
        self.delta_tol = delta_tol
        self.metric_precision = dict(metric_precision or {})
        self.min_val_width = min_val_width
        self.key_priority = key_priority or []

        self._prev: Optional[Dict[str, Any]] = None
        self._last_height: int = 0  # how many lines we printed last time

    # ---------- public API ----------
    def update(self, data: Dict[str, Any]) -> None:
        """Print (or reprint) the table for `data`, with diffs vs. prior call."""
        if not data:
            return

        grouped = self._group_by_namespace(data)
        # Order sections
        ns_names = list(grouped.keys())
        if self.fixed_section_order:
            pref = [ns for ns in self.fixed_section_order if ns in grouped]
            rest = sorted([ns for ns in ns_names if ns not in self.fixed_section_order])
            ns_order = pref + rest
        else:
            ns_order = sorted(ns_names)

        # Possibly sort keys within each section
        for ns in ns_order:
            if self.sort_keys_within_section:
                grouped[ns] = dict(sorted(grouped[ns].items(), key=lambda kv: self._get_sort_key(ns, kv[0])))

        # Build formatted values (including delta strings), then compute widths
        formatted = {}
        val_candidates = []
        key_candidates = [ns + "/" for ns in ns_order]
        for ns in ns_order:
            subdict = grouped[ns]
            f_sub = {}
            for sub, v in subdict.items():
                key_candidates.append(sub)

                # Compose value string: value + optional colored delta
                full_key = f"{ns}/{sub}" if sub else ns
                val_str = self._format_value(v, full_key)
                delta_str, color_name = self._delta_for_key(ns, sub, v)

                if delta_str:
                    # keep a small space before delta
                    delta_disp = _ansi(color_name, delta_str, self.color)
                    val_disp = f"{val_str} {delta_disp}"
                else:
                    val_disp = val_str

                f_sub[sub] = val_disp
                val_candidates.append(self._strip_ansi(val_disp))
            formatted[ns] = f_sub

        # Compute widths (include headers and values)
        indent = self.indent
        key_width = max(len(k) for k in key_candidates) if key_candidates else 0
        val_width = max(len(v) for v in val_candidates) if val_candidates else 0
        val_width = max(val_width, self.min_val_width)  # Apply minimum width

        # Row layout: "| " + (indent + key_width) + " | " + (val_width) + " |"
        border_len = 2 + (indent + key_width) + 3 + val_width + 2
        border = "-" * border_len

        # Prepare lines to print
        lines = []
        lines.append(border)
        for ns in ns_order:
            # Skip empty sections (no metrics under this namespace)
            if not formatted.get(ns):
                continue
            header = ns + "/"
            # Header: flush-left (no indent visually), but occupy same key field width
            header_line = f"| {header:<{indent + key_width}} | {'':>{val_width}} |"
            lines.append(_ansi("bold", header_line, self.color))
            for sub, val in formatted[ns].items():
                sub_disp = sub
                # Keep indent for metrics
                # Handle ANSI codes in val by manually calculating padding
                val_display_len = len(self._strip_ansi(val))
                val_padding = val_width - val_display_len
                val_padded = " " * val_padding + val if val_padding > 0 else val
                lines.append(f"| {' ' * indent}{sub_disp:<{key_width}} | {val_padded} |")
        lines.append(border)

        # Print the lines (either in-place or full redraw based on configuration)
        self._render_lines(lines)

        # Save state
        self._prev = dict(data)
        self._last_height = len(lines)

    # ---------- internals ----------
    def _render_lines(self, lines):
        text = "\n".join(lines)
        if self.use_ansi_inplace and self._last_height > 0:
            # Move cursor up by the last height, erase to end, then print
            # ESC[{n}F would move to beginning of line n lines up; ESC[{n}A moves cursor up.
            self.stream.write(f"\x1b[{self._last_height}F")  # move up N lines to the top border
            self.stream.write("\x1b[0J")                    # clear from cursor to end of screen
            self.stream.write(text + "\n")
            self.stream.flush()
        else:
            # First render or no ANSI — fall back to clear or normal print
            if not self.use_ansi_inplace:
                # optional: minimal clear to avoid scroll (comment out if undesired)
                os.system('cls' if os.name == 'nt' else 'clear')
            print(text, file=self.stream)

    def _group_by_namespace(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if "/" in k:
                ns, sub = k.split("/", 1)
            else:
                ns, sub = k, ""
            grouped.setdefault(ns, {})[sub] = v
        return grouped

    def _format_value(self, v: Any, full_key: str = "") -> str:
        if v is None:
            return "—"
        
        # Check if we have specific precision for this metric
        if full_key in self.metric_precision:
            precision = self.metric_precision[full_key]
            if _is_number(v):
                if precision == 0:  # Integer formatting
                    return str(int(round(v)))
                else:  # Float with specific precision
                    return f"{v:.{precision}f}"
        
        # Fall back to original logic
        if self.compact_numbers and _is_number(v):
            return _humanize_num(v, self.float_fmt)
        return _fmt_plain(v, self.float_fmt)

    def _delta_for_key(self, ns: str, sub: str, v: Any):
        """Return (delta_str, color_name) for the full key vs previous."""
        if self._prev is None:
            return ("", None)
        full_key = f"{ns}/{sub}" if sub else ns
        if full_key not in self._prev:
            return ("", None)
        prev_v = self._prev[full_key]
        if not (_is_number(v) and _is_number(prev_v)):
            return ("", None)
        delta = v - prev_v
        if abs(delta) <= self.delta_tol:
            return ("→0", "gray")  # unchanged / negligible
        # Improvement logic:
        color = None
        arrow = "↑" if delta > 0 else "↓"
        if full_key in self.better_when_increasing:
            inc_better = self.better_when_increasing[full_key]
            improved = (delta > 0) if inc_better else (delta < 0)
            color = "green" if improved else "red"
        else:
            # If no preference, color by sign
            color = "green" if delta > 0 else "red"
        # Format delta magnitude using per-metric precision when available, with dynamic fallback
        mag = self._format_delta_magnitude(abs(delta), full_key)
        return (f"{arrow}{mag}", color)

    def _format_delta_magnitude(self, delta: numbers.Number, full_key: str) -> str:
        """Format delta magnitude so small but real changes don't round to 0.

        Rules:
        - For integer-like values, reuse compact human formatting (k/M/B) when enabled.
        - For floats, prefer metric-specific precision (if provided via metric_precision).
          If that precision would round to 0, increase precision by 2 decimals.
          If still 0, fall back to scientific notation.
        """
        # Integers keep current compact behavior
        if isinstance(delta, int):
            return _humanize_num(delta, self.float_fmt) if self.compact_numbers else _fmt_plain(delta, self.float_fmt)

        # Some libraries may give numpy scalars; normalize types
        try:
            import numpy as _np  # local import to avoid top-level dependency
            if isinstance(delta, (_np.generic,)):
                delta = delta.item()
        except Exception:
            pass

        # Float formatting path
        if isinstance(delta, float):
            # Determine desired precision for this metric (value precision as proxy)
            precision = self.metric_precision.get(full_key)

            # Extract default decimals from self.float_fmt (e.g. ".2f" -> 2)
            def _decimals_from_fmt(fmt: str) -> int:
                try:
                    if fmt and fmt.startswith(".") and fmt.endswith("f"):
                        return int(fmt[1:-1])
                except Exception:
                    pass
                return 2

            default_decimals = _decimals_from_fmt(self.float_fmt)
            decimals = int(precision) if isinstance(precision, int) and precision >= 0 else default_decimals

            # First attempt with chosen decimals
            first = f"{delta:.{decimals}f}"
            if first.strip("0").strip(".") != "":
                # If compact requested and number is large enough, keep humanized style
                try:
                    as_float = float(first)
                except ValueError:
                    as_float = delta
                if self.compact_numbers and abs(as_float) >= 1000:
                    return _humanize_num(as_float, self.float_fmt)
                return first

            # If it still rounds to 0, increase precision
            more_decimals = decimals + 2
            second = f"{delta:.{more_decimals}f}"
            if second.strip("0").strip(".") != "":
                return second

            # Last resort: scientific notation
            return f"{delta:.2e}"

        # Fallback for other numeric types
        return _fmt_plain(delta, self.float_fmt)

    def _get_sort_key(self, namespace: str, subkey: str) -> tuple:
        """Generate a sort key that prioritizes specified keys, then alphabetical order."""
        full_key = f"{namespace}/{subkey}" if subkey else namespace
        
        # Check if this key is in our priority list
        try:
            priority_index = self.key_priority.index(full_key)
            return (0, priority_index)  # Priority items come first (0), then by their order
        except ValueError:
            # Not in priority list, sort alphabetically after priority items
            return (1, subkey.lower())  # Non-priority items come second (1), then alphabetically

    @staticmethod
    def _strip_ansi(s: str) -> str:
        # cheap and cheerful: remove \x1b[...m sequences for width calc
        out = []
        i = 0
        while i < len(s):
            if s[i] == "\x1b":
                # skip until 'm' or end
                while i < len(s) and s[i] != "m":
                    i += 1
                if i < len(s):
                    i += 1
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

# ========= Optional: backwards-compatible function =========

# Create a default singleton so you can keep calling print_namespaced_dict(...)
_default_printer = NamespaceTablePrinter()

def print_namespaced_dict(
    data: Dict[str, Any],
    *,
    inplace: bool = False,
    float_fmt: str = ".2f",
    compact_numbers: bool = True,
    color: bool = True,
    metric_precision: Optional[Dict[str, int]] = None,
    min_val_width: int = 15,
    key_priority: Optional[List[str]] = None,
):
    """
    Thin wrapper to keep your old callsite working. For advanced config,
    instantiate NamespaceTablePrinter yourself.
    
    Args:
        data: Dictionary with potentially namespaced keys
        inplace: Whether to update the display in-place
        float_fmt: Default float formatting
        compact_numbers: Whether to use compact number formatting
        color: Whether to use colored output
        metric_precision: Optional dict mapping metric keys to precision values.
                         0 means integer formatting, positive values specify decimal places.
        min_val_width: Minimum width for the values column
        key_priority: Optional list of keys to prioritize in sorting order
    """
    # If the user changes core options, recreate the singleton
    global _default_printer
    if (
        _default_printer.float_fmt != float_fmt
        or _default_printer.compact_numbers != compact_numbers
        or (_default_printer.use_ansi_inplace != bool(inplace and sys.stdout.isatty()))
        or _default_printer.color != bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        or _default_printer.metric_precision != (metric_precision or {})
        or _default_printer.min_val_width != min_val_width
        or _default_printer.key_priority != (key_priority or [])
    ):
        _default_printer = NamespaceTablePrinter(
            float_fmt=float_fmt,
            compact_numbers=compact_numbers,
            use_ansi_inplace=bool(inplace and sys.stdout.isatty()),
            color=bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None),
            metric_precision=metric_precision,
            min_val_width=min_val_width,
            key_priority=key_priority,
        )
    _default_printer.update(data)