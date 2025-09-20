import math
from typing import Any, Dict, Iterable, List, Optional

from utils.logging import format_section_footer, format_section_header


def downsample(seq: Iterable[float], target: int) -> List[float]:
    """Uniformly downsample a sequence to a target length (best-effort).

    Keeps the first element and samples by index stride; returns a new list.
    """
    seq = list(seq)
    if target <= 0 or len(seq) <= target:
        return list(seq)
    step = len(seq) / float(target)
    return [seq[int(i * step)] for i in range(target)]


def sparkline(values: Iterable[float], width: int) -> str:
    """Return a Unicode sparkline for the given values at the desired width.

    Uses an 8-level block set; returns an empty string when insufficient data.
    """

    # If no values or width is less than or equal to 0, return an empty string
    blocks = "▁▂▃▄▅▆▇█"
    values = list(values)
    if not values or width <= 0:
        return ""

    # Compute range using only finite values to avoid NaNs from inf/-inf
    finite_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite_values:
        # Nothing renderable
        return ""

    vmin = min(finite_values)
    vmax = max(finite_values)
    if vmax == vmin:
        # Flat line when all finite values are equal
        return "─" * max(1, min(width, len(values)))

    # Downsample to target width for rendering
    data = downsample(values, max(1, width))
    out: List[str] = []
    rng = vmax - vmin
    if rng <= 0 or not math.isfinite(rng):
        rng = 1.0

    # Map value to block index, handling non-finite gracefully
    for v in data:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            # Map -inf to lowest block, +inf to highest block, NaN to space
            if isinstance(v, float) and math.isinf(v):
                idx = 0 if v < 0 else (len(blocks) - 1)
                out.append(blocks[idx])
            else:
                out.append(" ")
            continue

        # Normalize and clamp
        norm = (v - vmin) / rng
        if not math.isfinite(norm):
            out.append(" ")
            continue
        norm = max(0.0, min(1.0, norm))
        idx = int(norm * (len(blocks) - 1))
        out.append(blocks[idx])

    return "".join(out)


def print_terminal_ascii_summary(history, max_metrics: int = 50, width: int = 48, per_metric_cap: int = 2000):
    """Print a compact ASCII sparkline summary of numeric metrics."""

    # Prefer train/* then eval/* then others for readability
    keys = sorted(history.keys(), key=lambda k: (0 if k.startswith("train/") else 1 if k.startswith("val/") else 2, k))
    shown = 0
    printed_header = False
    for k in keys:
        pts = history.get(k) or []

        # Show metrics even with a single point (e.g., val/* at end-of-run)
        if len(pts) < 1: 
            continue

        # Collapse duplicate steps (keep last)
        by_step = {}
        for s, v in pts:
            by_step[int(s)] = float(v)

        # If there are no points, skip
        if not by_step: continue

        # Sort the points by step
        steps_sorted = sorted(by_step)
        values = [by_step[s] for s in steps_sorted]
        # Build a chart and render it in a fixed-width column so that
        # subsequent stats align across metrics. For short histories
        # (e.g., val/* with one point), right-align the chart to leave
        # "past" entries visually empty on the left.
        raw_chart = sparkline(values, width)
        chart = raw_chart.rjust(max(0, int(width)))

        # Calculate the minimum, maximum, mean, and standard deviation
        vmin = min(values)
        vmax = max(values)
        vmean = sum(values) / len(values)

        # Population standard deviation (stable for streams); avoid tiny negatives from fp error
        if len(values) > 1:
            mean = vmean
            var = sum((x - mean) ** 2 for x in values) / len(values)
            vstd = var ** 0.5
        else:
            vstd = 0.0

        # Get the last value
        vlast = values[-1]
        if not printed_header:
            # Use shared header/footer formatting for consistency
            print("\n" + format_section_header("Metric History".upper(), width=width))
            printed_header = True
        
        # Pipe-separated stats for easier parsing/reading
        print(
            f"{k:>26}: {chart} | min={vmin:.4g} | max={vmax:.4g} | mean={vmean:.4g} | std={vstd:.4g} | last={vlast:.4g}"
        )
        shown += 1

        # If we've shown the maximum number of metrics, break
        if shown >= max_metrics: break

    # If we've printed the header, print a message if there are no numeric metrics to summarize
    if printed_header:
        if shown == 0:
            print("(no numeric metrics to summarize)")
        print(format_section_footer(width=width))

def print_terminal_ascii_alerts(
    freq_alerts: List[Dict[str, Any]],
    total_epochs: Optional[int] = None,
    width: int = 48,
) -> None:
    """Print an ASCII alert summary."""
    print("\n" + format_section_header("Metric Alerts".upper(), width=width))
    for freq_alert in freq_alerts:
        alert = freq_alert["alert"]
        epoch_count = freq_alert.get("epoch_count", freq_alert.get("count", 0))
        total = total_epochs or 0
        frequency_repr: str
        if total > 0:
            percentage = (epoch_count / total) * 100
            percent_str = (
                f"{int(percentage)}%"
                if percentage.is_integer()
                else f"{percentage:.1f}%"
            )
            frequency_repr = f"{epoch_count}/{total} ({percent_str})"
        else:
            frequency_repr = str(epoch_count)
        print(f"\n- `{alert._id}` triggered in `{frequency_repr}` epochs of training:")
        print(f"  - message: {alert.message}") 
        print(f"  - tip: {alert.tip}")
    print(format_section_footer(width=width))
