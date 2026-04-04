"""Training completion reporting helpers."""

from __future__ import annotations

from typing import Optional

from utils.formatting import format_duration


def extract_elapsed_seconds(agent) -> Optional[float]:
    """Return elapsed seconds from an agent, if available."""
    cached = getattr(agent, "_fit_elapsed_seconds", None)
    if cached is not None:
        try:
            return float(cached)
        except (TypeError, ValueError):
            return None

    timings = getattr(agent, "timings", None)
    if timings is None or not hasattr(timings, "seconds_since"):
        return None
    try:
        return float(timings.seconds_since("on_fit_start"))
    except (TypeError, ValueError, KeyError, RuntimeError):
        return None


def print_training_completion(agent) -> None:
    """Print the final standardized completion message for a training run."""
    elapsed_seconds = extract_elapsed_seconds(agent)
    human = format_duration(elapsed_seconds) if isinstance(elapsed_seconds, (int, float)) else "unknown"
    reason = (
        getattr(agent, "_final_stop_reason", None)
        or getattr(agent, "_early_stop_reason", None)
        or "completed."
    )
    if isinstance(reason, str) and reason and not reason.endswith(".") and reason != "completed.":
        reason = f"{reason}."
    print(f"Training completed in {human}. Reason: {reason}")
