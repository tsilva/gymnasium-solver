"""Forward-only helpers for policies with forward(obs)->(dist, optional value)."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution, Categorical, Bernoulli, Independent

from .distributions import MaskedCategorical


@torch.inference_mode()
def policy_act(
    model: torch.nn.Module,
    obs: Tensor,
    *,
    deterministic: bool = False,
    return_dist: bool = False,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Distribution]]:
    """Compute action, log-prob, and value via forward(); zeros when value head absent."""

    dist, value = model(obs)
    if deterministic: actions = dist.mode
    else: actions = dist.sample()
    logprobs = dist.log_prob(actions)
    if value is None:
        value_out = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)
    else:
        value_out = value.squeeze(-1)
    if return_dist:
        return actions, logprobs, value_out, dist
    return actions, logprobs, value_out

@torch.inference_mode()
def policy_predict_values(model: torch.nn.Module, obs: Tensor) -> Tensor:
    """Return value predictions using forward(); zeros when absent."""
    _, value = model(obs)
    if value is None: return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    return value.squeeze(-1)


def create_action_distribution(
    logits: torch.Tensor,
    valid_actions: list[int] | None,
    action_space_type: str = "discrete"
) -> Distribution:
    """Create action distribution with optional masking.

    Args:
        logits: Raw logits from policy head
        valid_actions: Valid action indices (None for no masking)
        action_space_type: "discrete" or "multibinary"

    Returns:
        torch.distributions.Distribution
    """
    assert action_space_type in ("discrete", "multibinary"), \
        f"action_space_type must be 'discrete' or 'multibinary', got {action_space_type}"

    # Apply action masking if specified
    if valid_actions is not None:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, valid_actions] = False
        logits = logits.masked_fill(mask, float('-inf'))

    # Create distribution
    if action_space_type == "multibinary":
        probs = torch.sigmoid(logits)
        return Independent(Bernoulli(probs=probs), 1)
    elif valid_actions is not None:
        return MaskedCategorical(logits=logits)
    else:
        return Categorical(logits=logits)


