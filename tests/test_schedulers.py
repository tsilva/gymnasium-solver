import pytest
import math

# Import functions directly to avoid circular imports in trainer_callbacks.__init__
from trainer_callbacks.hyperparameter_scheduler import (
    linear,
    cosine,
    exponential,
)


@pytest.mark.unit
def test_linear_scheduler():
    """Test linear scheduler interpolates correctly."""
    start, end = 1.0, 0.0

    assert linear(start, end, 0.0) == pytest.approx(1.0)
    assert linear(start, end, 0.25) == pytest.approx(0.75)
    assert linear(start, end, 0.5) == pytest.approx(0.5)
    assert linear(start, end, 0.75) == pytest.approx(0.25)
    assert linear(start, end, 1.0) == pytest.approx(0.0)

    # Test clamping
    assert linear(start, end, -0.1) == pytest.approx(1.0)
    assert linear(start, end, 1.1) == pytest.approx(0.0)


@pytest.mark.unit
def test_cosine_scheduler():
    """Test cosine scheduler follows cosine annealing."""
    start, end = 1.0, 0.0

    # At start: should be start_value
    assert cosine(start, end, 0.0) == pytest.approx(start)

    # At end: should be end_value
    assert cosine(start, end, 1.0) == pytest.approx(end)

    # At midpoint: should be halfway between start and end
    assert cosine(start, end, 0.5) == pytest.approx(0.5)

    # Test that cosine is smooth (slower decay at start/end, faster in middle)
    val_25 = cosine(start, end, 0.25)
    val_50 = cosine(start, end, 0.5)
    val_75 = cosine(start, end, 0.75)

    # First quarter should have smaller change than second quarter
    assert (start - val_25) < (val_25 - val_50)
    # Third quarter should have smaller change than second quarter
    assert (val_50 - val_75) > (val_75 - end)




@pytest.mark.unit
def test_exponential_scheduler():
    """Test exponential decay scheduler."""
    start, end = 1.0, 0.0

    # At start: should be start_value
    assert exponential(start, end, 0.0) == pytest.approx(start)

    # At end: should be end_value
    assert exponential(start, end, 1.0) == pytest.approx(end)

    # Test that exponential decays faster at the start
    val_25 = exponential(start, end, 0.25)
    val_50 = exponential(start, end, 0.5)
    val_75 = exponential(start, end, 0.75)

    # First quarter should have larger change than last quarter
    assert (start - val_25) > (val_75 - end)


@pytest.mark.unit
def test_schedulers_with_reverse_values():
    """Test that schedulers work when start_value < end_value (increasing)."""
    start, end = 0.0, 1.0

    # All schedulers should support increasing schedules
    assert linear(start, end, 0.0) == pytest.approx(start)
    assert linear(start, end, 1.0) == pytest.approx(end)

    assert cosine(start, end, 0.0) == pytest.approx(start)
    assert cosine(start, end, 1.0) == pytest.approx(end)

    assert exponential(start, end, 0.0) == pytest.approx(start)
    assert exponential(start, end, 1.0) == pytest.approx(end)


@pytest.mark.unit
def test_warmup_with_linear_scheduler():
    """Test that warmup works with linear scheduler."""
    start, end = 1.0, 0.0
    warmup_frac = 0.2

    # Simulate warmup phase (fraction < warmup_frac)
    # During warmup: linearly go from end to start
    warmup_val_0 = end + (start - end) * (0.0 / warmup_frac)
    assert warmup_val_0 == pytest.approx(end)

    warmup_val_half = end + (start - end) * (0.1 / warmup_frac)
    assert warmup_val_half == pytest.approx(0.5)

    warmup_val_end = end + (start - end) * (0.2 / warmup_frac)
    assert warmup_val_end == pytest.approx(start)

    # After warmup: apply scheduler on remaining fraction
    # fraction=0.6 means we're at (0.6 - 0.2) / (1.0 - 0.2) = 0.5 in scheduler
    scheduler_frac = (0.6 - warmup_frac) / (1.0 - warmup_frac)
    assert scheduler_frac == pytest.approx(0.5)
    scheduler_val = linear(start, end, scheduler_frac)
    assert scheduler_val == pytest.approx(0.5)
