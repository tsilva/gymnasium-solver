
def print_terminal_ascii_summary(history, max_metrics: int = 50, width: int = 48, per_metric_cap: int = 2000):
    """Print an ASCII sparkline summary of recorded numeric metrics.

    Args:
        max_metrics: Maximum number of metrics to print to avoid long outputs.
        width: Target width of sparklines.
        per_metric_cap: Safety cap (ignored here but kept for future trimming consistency).
    """

    def downsample(seq, target):
        if len(seq) <= target:
            return seq
        # Uniform uniform sampling by index
        step = len(seq) / float(target)
        return [seq[int(i * step)] for i in range(target)]

    def spark(values, w):
        blocks = "▁▂▃▄▅▆▇█"
        if not values:
            return ""
        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            return "─" * max(1, min(w, len(values)))
        data = downsample(values, max(1, w))
        out = []
        rng = (vmax - vmin) or 1.0
        for v in data:
            idx = int((v - vmin) / rng * (len(blocks) - 1))
            out.append(blocks[max(0, min(idx, len(blocks) - 1))])
        return "".join(out)

    # Prefer train/* then eval/* then others for readability
    keys = sorted(history.keys(), key=lambda k: (0 if k.startswith("train/") else 1 if k.startswith("eval/") else 2, k))
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
        chart = spark(values, width)
        vmin = min(values)
        vmax = max(values)
        vlast = values[-1]
        if not printed_header:
            print("\n=== Metrics Summary (ASCII) ===")
            printed_header = True
        print(f"{k:>26}: {chart}  min={vmin:.4g} max={vmax:.4g} last={vlast:.4g}")
        shown += 1
        if shown >= max_metrics:
            break
    if printed_header:
        if shown == 0:
            print("(no numeric metrics to summarize)")
        print("=" * 30)