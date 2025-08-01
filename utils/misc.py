import torch
import itertools
from typing import Dict, Any
from contextlib import contextmanager
import os
import sys
from typing import Dict, Any, Iterable, Optional
import numbers
import torch
from torch.utils.data import DataLoader, TensorDataset

def create_dummy_dataloader(n_samples: int = 1, sample_dim: int = 1, batch_size: int = 1) -> torch.utils.data.DataLoader:
    dummy_data = torch.zeros(n_samples, sample_dim)
    dummy_target = torch.zeros(n_samples, sample_dim)
    dataset = TensorDataset(dummy_data, dummy_target)
    return DataLoader(dataset, batch_size=batch_size)

def prefix_dict_keys(data: dict, prefix: str) -> dict:
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}

def print_namespaced_dict(data: dict, metric_precision: Optional[Dict[str, int]] = None) -> None:
    """
    Prints a dictionary with namespaced keys (e.g., 'rollout/ep_len_mean')
    in a formatted ASCII table grouped by namespaces.
    
    Args:
        data: Dictionary with potentially namespaced keys
        metric_precision: Optional dict mapping metric keys to precision values.
                         0 means integer formatting, positive values specify decimal places.
                         E.g., {"train/loss": 4, "rollout/ep_len_mean": 0}
    """
    if not data: return
    metric_precision = metric_precision or {}
    
    # Group keys by their namespace prefix
    grouped = {}
    for key, value in data.items():
        if "/" in key:
            namespace, subkey = key.split("/", 1)
        else:
            namespace, subkey = key, ""
        grouped.setdefault(namespace, {})[subkey] = value

    # Format values with custom precision if specified
    formatted_grouped = {}
    for ns, subdict in grouped.items():
        formatted_grouped[ns] = {}
        for subkey, val in subdict.items():
            full_key = f"{ns}/{subkey}" if subkey else ns
            
            if full_key in metric_precision and _is_number(val):
                precision = metric_precision[full_key]
                if precision == 0:  # Integer formatting
                    formatted_val = str(int(round(val)))
                else:  # Float with specific precision
                    formatted_val = f"{val:.{precision}f}"
            else:
                formatted_val = str(val)
                
            formatted_grouped[ns][subkey] = formatted_val

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
        fixed_section_order: Optional[Iterable[str]] = ("train", "time", "rollout"),
        sort_keys_within_section: bool = True,
        # Render method
        use_ansi_inplace: bool = True,
        stream=None,
        delta_tol: float = 1e-12,
        # Precision per metric: {"namespace/subkey": precision, ...} where 0 = int
        metric_precision: Optional[Dict[str, int]] = None,
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
                grouped[ns] = dict(sorted(grouped[ns].items(), key=lambda kv: kv[0]))

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

        # Row layout: "| " + (indent + key_width) + " | " + (val_width) + " |"
        border_len = 2 + (indent + key_width) + 3 + val_width + 2
        border = "-" * border_len

        # Prepare lines to print
        lines = []
        lines.append(border)
        for ns in ns_order:
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

        # In-place reprint without flicker
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
        # format delta magnitude compactly
        mag = _humanize_num(abs(delta), self.float_fmt) if self.compact_numbers else _fmt_plain(abs(delta), self.float_fmt)
        return (f"{arrow}{mag}", color)

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
    """
    # If the user changes core options, recreate the singleton
    global _default_printer
    if (
        _default_printer.float_fmt != float_fmt
        or _default_printer.compact_numbers != compact_numbers
        or (_default_printer.use_ansi_inplace != bool(inplace and sys.stdout.isatty()))
        or _default_printer.color != bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        or _default_printer.metric_precision != (metric_precision or {})
    ):
        _default_printer = NamespaceTablePrinter(
            float_fmt=float_fmt,
            compact_numbers=compact_numbers,
            use_ansi_inplace=bool(inplace and sys.stdout.isatty()),
            color=bool(color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None),
            metric_precision=metric_precision,
        )
    _default_printer.update(data)


# metrics_table_callback.py
from typing import Iterable, Optional, Dict, Any
import re
import torch
import pytorch_lightning as pl

class StdoutMetricsTable(pl.Callback):
    """
    Periodically prints a table to stdout with the *latest value* for each logged metric.

    Sources:
      - trainer.logged_metrics        (step-level latest values)
      - trainer.callback_metrics      (epoch-level aggregated values)
      - trainer.progress_bar_metrics  (if present; UI-focused values)

    Usage:
      printer = StdoutMetricsTable(every_n_steps=200, every_n_epochs=1, include=[r'^train/', r'^val/'])
      trainer = pl.Trainer(callbacks=[printer], ...)
    """

    def __init__(
        self,
        every_n_steps: Optional[int] = None,   # if None, don't print by step
        every_n_epochs: Optional[int] = 1,     # print each epoch by default
        include: Optional[Iterable[str]] = None,  # regex patterns to keep
        exclude: Optional[Iterable[str]] = None,  # regex patterns to drop
        digits: int = 4,                          # rounding for floats
        metric_precision: Optional[Dict[str, int]] = None,  # precision per metric
    ):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.include = [re.compile(p) for p in (include or [])]
        self.exclude = [re.compile(p) for p in (exclude or [])]
        self.digits = digits
        self.metric_precision = metric_precision or {}

    # ---------- hooks ----------
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx):
        if self.every_n_steps is None:
            return
        # global_step increments after optimizer step; print when divisible
        if trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            self._maybe_print(trainer, stage="train-step")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_print(trainer, stage="val-epoch")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.every_n_epochs is not None and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self._maybe_print(trainer, stage="train-epoch")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Always print at test end if enabled by epoch cadence
        if self.every_n_epochs is not None:
            self._maybe_print(trainer, stage="test-epoch")

    # ---------- internals ----------
    def _maybe_print(self, trainer: "pl.Trainer", stage: str):
        # Only the main process prints
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        metrics = self._collect_metrics(trainer)
        metrics = self._filter_metrics(metrics)
        if not metrics:
            return

        step = getattr(trainer, "global_step", None)
        epoch = getattr(trainer, "current_epoch", None)
        header = f"[{stage}] epoch={epoch} step={step}"
        self._print_table(metrics, header)

    def _collect_metrics(self, trainer: "pl.Trainer") -> Dict[str, Any]:
        combo: Dict[str, Any] = {}

        # Merge, with later sources taking precedence
        dicts = []
        if hasattr(trainer, "logged_metrics"):
            dicts.append(trainer.logged_metrics)
        if hasattr(trainer, "callback_metrics"):
            dicts.append(trainer.callback_metrics)
        if hasattr(trainer, "progress_bar_metrics"):  # not always present
            dicts.append(trainer.progress_bar_metrics)

        for d in dicts:
            for k, v in d.items():
                combo[k] = self._to_python_scalar(v)
        # Drop common housekeeping keys if present
        for k in ["epoch", "step", "global_step"]:
            combo.pop(k, None)
        return combo

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self.include:
            metrics = {k: v for k, v in metrics.items() if any(p.search(k) for p in self.include)}
        if self.exclude:
            metrics = {k: v for k, v in metrics.items() if not any(p.search(k) for p in self.exclude)}
        return metrics

    def _to_python_scalar(self, x: Any) -> Any:
        # Convert tensors and numpy scalars to plain Python types when possible
        try:
            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return x.detach().item()
                return x.detach().float().mean().item()
            # numpy scalar support without importing numpy explicitly
            if hasattr(x, "item") and callable(getattr(x, "item")):
                return x.item()
            return x
        except Exception:
            return x

    def _print_table(self, metrics: Dict[str, Any], header: str):
        from utils.misc import print_namespaced_dict
        print_namespaced_dict(metrics, metric_precision=self.metric_precision)

    def _format_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.digits}f}"
        return str(v)