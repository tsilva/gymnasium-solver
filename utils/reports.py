def print_terminal_ascii_summary(history, max_metrics: int = 50, width: int = 48, per_metric_cap: int = 2000):
    """Print an ASCII sparkline summary of recorded numeric metrics.

    Args:
        max_metrics: Maximum number of metrics to print to avoid long outputs.
        width: Target width of sparklines.
        per_metric_cap: Safety cap (ignored here but kept for future trimming consistency).
    """
    # Local import to avoid heavier dependencies at import-time
    try:
        from utils.metrics import get_key_priority
        KEY_PRIORITY = get_key_priority() or []
    except Exception:
        KEY_PRIORITY = []

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

    # Build stable order consistent with training table printer:
    # 1) Group by namespace (train, eval first, then others alpha)
    # 2) Within each namespace, order by key_priority, then alpha
    def _split_ns(key: str):
        if "/" in key:
            ns, sub = key.split("/", 1)
            return ns, sub
        return "", key

    # Namespace order: train, eval, then alphabetical of remaining namespaces (including empty "")
    namespaces = {}
    for k in history.keys():
        ns, sub = _split_ns(k)
        namespaces.setdefault(ns, set()).add(sub)

    fixed_first = [ns for ns in ("train", "eval") if ns in namespaces]
    remaining = sorted([ns for ns in namespaces.keys() if ns not in ("train", "eval")])
    ns_order = fixed_first + remaining

    # Sorting helper within a namespace
    def _sort_key(ns: str, sub: str):
        full = f"{ns}/{sub}" if ns else sub
        try:
            idx = KEY_PRIORITY.index(full)
            return (0, idx)
        except ValueError:
            # Fall back to alpha by sub-key (case-insensitive)
            return (1, sub.lower())

    ordered_keys = []
    for ns in ns_order:
        subs = sorted(list(namespaces[ns]), key=lambda s: _sort_key(ns, s))
        for sub in subs:
            full = f"{ns}/{sub}" if ns else sub
            if full in history:
                ordered_keys.append(full)
    shown = 0
    printed_header = False
    for k in ordered_keys:
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