"""Forward-only helpers for policies with forward(obs)->(dist, optional value)."""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Distribution


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

 
