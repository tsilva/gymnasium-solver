from typing import Iterable, List


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
    blocks = "▁▂▃▄▅▆▇█"
    values = list(values)
    if not values or width <= 0:
        return ""
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return "─" * max(1, min(width, len(values)))
    data = downsample(values, max(1, width))
    out = []
    rng = (vmax - vmin) or 1.0
    for v in data:
        idx = int((v - vmin) / rng * (len(blocks) - 1))
        out.append(blocks[max(0, min(idx, len(blocks) - 1))])
    return "".join(out)


def print_terminal_ascii_summary(history, max_metrics: int = 50, width: int = 48, per_metric_cap: int = 2000):
    """Print an ASCII sparkline summary of recorded numeric metrics.

    Args:
        max_metrics: Maximum number of metrics to print to avoid long outputs.
        width: Target width of sparklines.
        per_metric_cap: Safety cap (ignored here but kept for future trimming consistency).
    """

    # local aliases for backward-compat names
    def spark(values, w):
        return sparkline(values, w)

    # Prefer train/* then eval/* then others for readability
    keys = sorted(history.keys(), key=lambda k: (0 if k.startswith("train/") else 1 if k.startswith("val/") else 2, k))
    shown = 0
    printed_header = False
    for k in keys:
        pts = history.get(k) or []
        if len(pts) < 2:
            continue
        # Collapse duplicate steps (keep last)
        by_step = {}
        for s, v in pts:
            by_step[int(s)] = float(v)
        if not by_step:
            continue
        steps_sorted = sorted(by_step)
        values = [by_step[s] for s in steps_sorted]
        chart = sparkline(values, width)
        vmin = min(values)
        vmax = max(values)
        vmean = sum(values) / len(values)
        # population std (stable for streams); avoid tiny negatives from fp error
        if len(values) > 1:
            mean = vmean
            var = sum((x - mean) ** 2 for x in values) / len(values)
            vstd = var ** 0.5
        else:
            vstd = 0.0
        vlast = values[-1]
        if not printed_header:
            print("\n=== Metrics Summary (ASCII) ===")
            printed_header = True
        # Pipe-separated stats for easier parsing/reading
        print(
            f"{k:>26}: {chart} | min={vmin:.4g} | max={vmax:.4g} | mean={vmean:.4g} | std={vstd:.4g} | last={vlast:.4g}"
        )
        shown += 1
        if shown >= max_metrics:
            break
    if printed_header:
        if shown == 0:
            print("(no numeric metrics to summarize)")
        print("=" * 30)
