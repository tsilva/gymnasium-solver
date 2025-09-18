"""Forward-only policy operations.

Provides small helpers that operate on policies that implement a single
forward(obs) -> (Distribution, Optional[value]) interface. Centralizing these
keeps model classes minimal and moves action/value logic out to call sites.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


@torch.inference_mode()
def policy_act(
    model: torch.nn.Module,
    obs: Tensor,
    *,
    deterministic: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute action, log-prob, and value using only forward().

    - Samples from the distribution unless deterministic=True, in which case
      it uses the distribution mode (argmax for Categorical).
    - If the model has no value head (returns None), a zero tensor is returned
      for the value prediction with shape (batch,).
    """

    dist, value = model(obs)
    if deterministic: actions = dist.mode
    else: actions = dist.sample()
    logprobs = dist.log_prob(actions)
    if value is None: value = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)
    return actions, logprobs, value.squeeze(-1)

@torch.inference_mode()
def policy_predict_values(model: torch.nn.Module, obs: Tensor) -> Tensor:
    """Return value predictions using forward(); zeros when absent."""
    _, value = model(obs)
    if value is None: return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    return value.squeeze(-1)

 
