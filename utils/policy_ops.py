"""Forward-only policy operations.

Provides small helpers that operate on policies that implement a single
forward(obs) -> (Distribution, Optional[value]) interface. Centralizing these
keeps model classes minimal and moves action/value logic out to call sites.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Categorical, Distribution


def policy_forward(model: torch.nn.Module, obs: Tensor) -> Tuple[Distribution, Optional[Tensor]]:
    """Call the model's forward and normalize the return.

    Expects models to return (Distribution, Optional[Tensor]). Value may be None
    for policy-only networks.
    """
    # Support plain objects exposing .forward without __call__
    if hasattr(model, "forward"):
        dist, value = model.forward(obs)  # type: ignore[attr-defined]
    else:
        dist, value = model(obs)
    return dist, value


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
    # Support alternate minimal policy API used in tests: act() / predict_values()
    if hasattr(model, "act"):
        try:
            a, logp, v = model.act(obs, deterministic=deterministic)  # type: ignore[attr-defined]
            if v is None:
                v = torch.zeros(a.shape[0], dtype=torch.float32, device=a.device)
            return a, logp, v.squeeze(-1)
        except TypeError:
            # Fallback to distribution path if signature mismatch
            pass

    dist, value = policy_forward(model, obs)
    if deterministic:
        # Categorical exposes .mode; for generic distributions, fallback to argmax over probs if present
        try:
            actions = dist.mode  # type: ignore[attr-defined]
        except Exception:
            # Best-effort fallback: treat as categorical over probs/logits
            if hasattr(dist, "probs") and isinstance(dist.probs, Tensor):  # type: ignore[attr-defined]
                actions = dist.probs.argmax(dim=-1)  # type: ignore[attr-defined]
            elif hasattr(dist, "logits") and isinstance(dist.logits, Tensor):  # type: ignore[attr-defined]
                actions = dist.logits.argmax(dim=-1)  # type: ignore[attr-defined]
            else:
                actions = dist.sample()
    else:
        actions = dist.sample()

    log_prob = dist.log_prob(actions)
    if value is None:
        value = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)
    return actions, log_prob, value.squeeze(-1)


@torch.inference_mode()
def policy_predict_values(model: torch.nn.Module, obs: Tensor) -> Tensor:
    """Return value predictions using forward(); zeros when absent."""
    # Support alternate minimal policy API
    if hasattr(model, "predict_values"):
        try:
            v = model.predict_values(obs)  # type: ignore[attr-defined]
            return v.squeeze(-1)
        except Exception:
            pass
    _, value = policy_forward(model, obs)
    if value is None:
        return torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    return value.squeeze(-1)


def policy_evaluate_actions(
    model: torch.nn.Module,
    obs: Tensor,
    actions: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return log_prob, entropy, and value predictions for given actions.

    Value is zeros when the model has no value head.
    """
    dist, value = policy_forward(model, obs)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()
    if value is None:
        value = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    return log_prob, entropy, value.squeeze(-1)
