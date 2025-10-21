"""Tests for MaskedCategorical distribution."""

import math
import pytest
import torch
from torch.distributions import Categorical

from utils.distributions import MaskedCategorical


@pytest.mark.unit
def test_masked_categorical_entropy_vs_standard():
    """Test that MaskedCategorical computes entropy correctly over valid actions only."""
    batch_size = 4
    n_actions = 5

    # Create uniform logits
    logits = torch.zeros(batch_size, n_actions)

    # For standard Categorical with uniform distribution, entropy = log(n_actions)
    standard_dist = Categorical(logits=logits)
    standard_entropy = standard_dist.entropy()
    expected_standard = math.log(n_actions)
    assert torch.allclose(standard_entropy, torch.tensor(expected_standard), atol=1e-5)

    # Mask out 2 actions (leaving 3 valid)
    # Set actions 3 and 4 to -inf
    masked_logits = logits.clone()
    masked_logits[:, [3, 4]] = float('-inf')

    # For MaskedCategorical with 3 valid actions, entropy should be log(3)
    masked_dist = MaskedCategorical(logits=masked_logits)
    masked_entropy = masked_dist.entropy()
    expected_masked = math.log(3)

    assert torch.allclose(masked_entropy, torch.tensor(expected_masked), atol=1e-5), \
        f"Expected entropy {expected_masked}, got {masked_entropy.mean().item()}"

    # Verify it's different from standard entropy
    assert not torch.allclose(masked_entropy, standard_entropy), \
        "Masked entropy should differ from standard entropy"


@pytest.mark.unit
def test_masked_categorical_entropy_single_valid_action():
    """Test that entropy is 0 when only one action is valid."""
    batch_size = 2
    n_actions = 5

    logits = torch.zeros(batch_size, n_actions)

    # Mask out all actions except action 0
    logits[:, 1:] = float('-inf')

    dist = MaskedCategorical(logits=logits)
    entropy = dist.entropy()

    # Entropy should be 0 (deterministic distribution)
    assert torch.allclose(entropy, torch.zeros_like(entropy), atol=1e-5), \
        f"Expected entropy 0 for single valid action, got {entropy}"


@pytest.mark.unit
def test_masked_categorical_entropy_non_uniform():
    """Test entropy calculation with non-uniform probabilities over valid actions."""
    batch_size = 1
    n_actions = 4

    # Create non-uniform logits: [2.0, 1.0, -inf, -inf]
    # This means only actions 0 and 1 are valid
    logits = torch.tensor([[2.0, 1.0, float('-inf'), float('-inf')]])

    dist = MaskedCategorical(logits=logits)
    entropy = dist.entropy()

    # Compute expected entropy manually
    # After softmax over valid actions: p0 = e^2/(e^2 + e^1), p1 = e^1/(e^2 + e^1)
    p0 = math.exp(2.0) / (math.exp(2.0) + math.exp(1.0))
    p1 = math.exp(1.0) / (math.exp(2.0) + math.exp(1.0))
    expected_entropy = -(p0 * math.log(p0) + p1 * math.log(p1))

    assert torch.allclose(entropy, torch.tensor(expected_entropy), atol=1e-5), \
        f"Expected entropy {expected_entropy}, got {entropy.item()}"


@pytest.mark.unit
def test_masked_categorical_log_prob():
    """Test that log_prob works correctly for masked distribution."""
    batch_size = 3
    n_actions = 5

    logits = torch.zeros(batch_size, n_actions)

    # Mask actions 3 and 4
    logits[:, [3, 4]] = float('-inf')

    dist = MaskedCategorical(logits=logits)

    # Sample valid actions
    valid_actions = torch.tensor([0, 1, 2])
    log_probs = dist.log_prob(valid_actions)

    # For uniform distribution over 3 actions, log_prob should be log(1/3) = -log(3)
    expected_log_prob = -math.log(3)
    assert torch.allclose(log_probs, torch.tensor(expected_log_prob), atol=1e-5), \
        f"Expected log_prob {expected_log_prob}, got {log_probs}"


@pytest.mark.unit
def test_masked_categorical_sampling():
    """Test that sampling never produces masked actions."""
    batch_size = 100
    n_actions = 10

    logits = torch.zeros(batch_size, n_actions)

    # Mask out actions 5-9 (only 0-4 are valid)
    logits[:, 5:] = float('-inf')
    valid_actions = set(range(5))

    dist = MaskedCategorical(logits=logits)

    # Sample many times
    for _ in range(10):
        samples = dist.sample()
        assert samples.shape == (batch_size,)
        # Verify all samples are in valid action set
        assert all(s.item() in valid_actions for s in samples), \
            f"Sampled invalid actions: {[s.item() for s in samples if s.item() not in valid_actions]}"


@pytest.mark.unit
def test_masked_categorical_probs_sum_to_one():
    """Test that probabilities sum to 1.0 after masking."""
    batch_size = 2
    n_actions = 6

    logits = torch.randn(batch_size, n_actions)

    # Mask out half the actions randomly
    logits[:, [1, 3, 5]] = float('-inf')

    dist = MaskedCategorical(logits=logits)
    probs = dist.probs

    # Sum should be 1.0
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), \
        f"Probabilities don't sum to 1: {prob_sums}"

    # Masked actions should have 0 probability
    assert torch.allclose(probs[:, [1, 3, 5]], torch.zeros(batch_size, 3), atol=1e-8), \
        "Masked actions should have 0 probability"


@pytest.mark.unit
def test_masked_categorical_backward_compatible():
    """Test that MaskedCategorical behaves like Categorical when no masking."""
    batch_size = 4
    n_actions = 5

    logits = torch.randn(batch_size, n_actions)

    standard_dist = Categorical(logits=logits)
    masked_dist = MaskedCategorical(logits=logits)

    # Entropy should match
    assert torch.allclose(standard_dist.entropy(), masked_dist.entropy(), atol=1e-5)

    # Log probs should match
    actions = torch.randint(0, n_actions, (batch_size,))
    assert torch.allclose(
        standard_dist.log_prob(actions),
        masked_dist.log_prob(actions),
        atol=1e-5
    )

    # Probs should match
    assert torch.allclose(standard_dist.probs, masked_dist.probs, atol=1e-5)
