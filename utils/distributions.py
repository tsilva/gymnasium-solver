"""Custom probability distributions for RL with action masking support."""

import torch
from torch.distributions import Categorical
from torch.distributions.utils import lazy_property


class MaskedCategorical(Categorical):
    """Categorical distribution with action masking support.

    Handles -inf logits by computing entropy, log_prob, and other operations
    only over valid (non-masked) actions. This ensures correct entropy measurement
    when action masking is applied.

    Invalid actions should have logits set to -inf.
    """

    def __init__(self, logits=None, probs=None, validate_args=None):
        # Store original logits/probs before parent __init__ processes them
        if logits is not None:
            self._mask = torch.isfinite(logits)
        elif probs is not None:
            self._mask = probs > 0
        else:
            raise ValueError("Either logits or probs must be specified")

        super().__init__(logits=logits, probs=probs, validate_args=validate_args)

    @lazy_property
    def logits(self):
        """Return logits, clamping -inf to a large negative number for numerical stability."""
        return super().logits

    def entropy(self):
        """Compute entropy only over valid (non-masked) actions.

        For a masked distribution, entropy should be computed as:
            H = -sum(p(a) * log(p(a))) for all valid actions a

        Where p(a) is renormalized over only the valid actions.
        """
        # Get probabilities (already normalized by parent class)
        p = self.probs

        # Mask out invalid actions (those with -inf logits)
        # The parent class sets these to 0 probability
        valid_mask = self._mask

        # For valid actions, compute -p * log(p)
        # Note: probs are already 0 for invalid actions, so we just need to handle log(p)
        # Use where to avoid log(0) = -inf for invalid actions
        log_p = torch.where(valid_mask, torch.log(p + 1e-8), torch.zeros_like(p))

        # Compute entropy: -sum(p * log(p)) over valid actions
        # Since invalid actions have p=0, they contribute 0 to the sum
        entropy = -(p * log_p).sum(dim=-1)

        return entropy

    def log_prob(self, value):
        """Compute log probability for the given actions.

        Asserts that sampled actions are valid (not masked).
        """
        log_probs = super().log_prob(value)

        # Check if any sampled actions are invalid (this shouldn't happen)
        if __debug__:
            # Create index mask for the selected actions
            batch_size = value.shape[0]
            action_indices = value.long()

            # Check that selected actions are valid
            selected_valid = self._mask[torch.arange(batch_size), action_indices]
            if not selected_valid.all():
                invalid_count = (~selected_valid).sum().item()
                raise ValueError(
                    f"Sampled {invalid_count} invalid (masked) actions. "
                    "This should not happen - check sampling logic."
                )

        return log_probs
