"""Torch utilities for modules, devices, and inference context."""

from __future__ import annotations

import itertools
import math
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


# Canonical activation mapping used across the codebase
ACTIVATION_MAPPING = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "identity": nn.Identity,
}


@contextmanager
def inference_ctx(*modules):
    """Temporarily set modules to eval and disable grads; restore training flags after."""
    # Filter out non-modules and Nones; flatten lists/tuples
    flat = [
        m
        for m in itertools.chain.from_iterable(
            (m if isinstance(m, (list, tuple)) else (m,)) for m in modules
        )
        if isinstance(m, nn.Module) and m is not None
    ]

    was_training = [m.training for m in flat]
    try:
        for m in flat:
            m.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, flag in zip(flat, was_training):
            if flag:  # only restore if it was in train mode
                m.train()


def _device_of(module: torch.nn.Module) -> torch.device:
    """Return the device of the first parameter of a module, or CPU if unknown."""
    if not hasattr(module, "parameters"):
        return torch.device("cpu")
    return next(module.parameters()).device


def _gain_for_activation_module(act: nn.Module) -> float | None:
    """Best-effort gain for an activation module; returns None if unknown."""
    if isinstance(act, nn.ReLU):
        return nn.init.calculate_gain("relu")
    if isinstance(act, nn.LeakyReLU):
        return nn.init.calculate_gain("leaky_relu", act.negative_slope)
    if isinstance(act, nn.Tanh):
        return nn.init.calculate_gain("tanh")
    if isinstance(act, nn.ELU):
        # ELU is not directly supported; ReLU gain is a reasonable proxy
        return nn.init.calculate_gain("relu")
    if isinstance(act, (nn.GELU, nn.SiLU)):
        # Common smooth ReLU-like activations
        return nn.init.calculate_gain("relu")
    if isinstance(act, (nn.SELU, nn.Identity)):
        return 1.0
    return None


def _activation_instance_from_spec(spec: "str | type[nn.Module] | nn.Module") -> nn.Module:
    if isinstance(spec, nn.Module):
        return spec
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec()
    if isinstance(spec, str):
        key = spec.lower()
        return ACTIVATION_MAPPING.get(key, nn.ReLU)()
    return nn.ReLU()


def assert_detached(*tensors: torch.Tensor) -> bool:
    """Assert tensors are detached from the computation graph."""
    for t in tensors:
        assert not t.requires_grad, "Tensor still requires grad"
        assert t.grad_fn is None, "Tensor is still connected to a computation graph"
    return True


def batch_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize tensor to zero mean and unit variance."""
    return (x - x.mean()) / (x.std() + eps)


def compute_kl_diagnostics(old_logprobs: torch.Tensor, new_logprobs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KL divergence diagnostics between old and new policy distributions.

    Args:
        old_logprobs: Log probabilities from the rollout policy
        new_logprobs: Log probabilities from the current policy

    Returns:
        kl_div: KL divergence (old_logprobs - new_logprobs).mean()
        approx_kl: Approximate KL divergence using ratio expansion
    """
    # Clamp log probability difference to prevent overflow in exp()
    # exp(20) ≈ 4.9e8, exp(-20) ≈ 2e-9, which covers typical policy shifts
    logprob_diff = torch.clamp(new_logprobs - old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(logprob_diff)
    kl_div = (old_logprobs - new_logprobs).mean()
    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
    return kl_div, approx_kl


def compute_kl_metrics(
    old_logprobs: torch.Tensor,
    new_logprobs: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute KL metrics for logging and return raw values.

    Args:
        old_logprobs: Log probabilities from the rollout policy
        new_logprobs: Log probabilities from the current policy

    Returns:
        Tuple of (metrics_dict, kl_div, approx_kl)
        - metrics_dict: Dictionary with 'opt/ppo/kl' and 'opt/ppo/approx_kl'
        - kl_div: Raw KL divergence tensor
        - approx_kl: Raw approximate KL tensor
    """
    with torch.no_grad():
        kl_div, approx_kl = compute_kl_diagnostics(old_logprobs, new_logprobs)

    metrics = {
        'opt/ppo/kl': kl_div.detach(),
        'opt/ppo/approx_kl': approx_kl.detach(),
    }
    return metrics, kl_div, approx_kl


def normalize_batch_with_metrics(
    tensor: torch.Tensor,
    normalize_mode: str | bool,
    metric_prefix: str
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalize tensor and return metrics if batch normalization enabled.

    Args:
        tensor: Tensor to normalize (advantages, returns, etc.)
        normalize_mode: One of "batch", "rollout", "off", False, or None
        metric_prefix: Metric namespace (e.g., "roll/adv")

    Returns:
        (normalized_tensor, metrics_dict)
    """
    assert normalize_mode in ("batch", "rollout", "off", "", False, None), \
        f"normalize_mode must be 'batch', 'rollout', 'off', False, or None, got {normalize_mode}"

    if normalize_mode != "batch":
        return tensor, {}

    normalized = batch_normalize(tensor)
    metrics = {
        f'{metric_prefix}/norm/mean': normalized.mean().detach(),
        f'{metric_prefix}/norm/std': normalized.std().detach()
    }
    return normalized, metrics


def compute_param_group_grad_norm(params):
    """Compute L2 grad norm over params; ignore None grads (0.0 if none)."""
    total_sq = 0.0
    has_grad = False
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        has_grad = True
        # Use .detach() to avoid graph tracking; flatten to 1D before norm
        total_sq += float(g.detach().data.norm(2).item() ** 2)
    if not has_grad:
        return 0.0
    return math.sqrt(total_sq)


def to_python_scalar(x: Any) -> Any:
    """Convert tensors/numpy scalars to Python scalars; averages multi-element tensors."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().item()
        return x.detach().float().mean().item()
    if hasattr(x, "item") and callable(getattr(x, "item")):
        return x.item()
    return x


def init_model_weights(
    model: nn.Module,
    *,
    default_activation: "str | type[nn.Module] | nn.Module" = nn.ReLU,
    policy_heads: "list[nn.Module] | tuple[nn.Module, ...] | None" = None,
    value_heads: "list[nn.Module] | tuple[nn.Module, ...] | None" = None,
) -> None:
    """Initialize model weights for RL: orthogonal layers, small policy head gain, unit value head gain."""

    policy_heads = tuple(policy_heads or ())
    value_heads = tuple(value_heads or ())

    # Pre-compute gains for layers that are directly followed by an activation
    # inside Sequentials (Conv/Linear then Activation pattern).
    gains_by_module: dict[nn.Module, float] = {}
    for module in model.modules():
        if isinstance(module, nn.Sequential):
            children = list(module.children())
            for i, child in enumerate(children):
                if isinstance(child, (nn.Linear, nn.Conv2d)):
                    next_mod = children[i + 1] if i + 1 < len(children) else None
                    if next_mod is not None:
                        gain = _gain_for_activation_module(next_mod)
                        if gain is not None:
                            gains_by_module[child] = gain

    default_act_inst = _activation_instance_from_spec(default_activation)
    default_gain = _gain_for_activation_module(default_act_inst) or 1.0

    def _init_linear_or_conv(layer: nn.Module, gain: float) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        elif isinstance(layer, nn.Conv2d):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    # First, initialize all layers with inferred/default gains
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if m in policy_heads or m in value_heads:
                # Defer heads to the next step
                continue
            gain = gains_by_module.get(m, default_gain)
            _init_linear_or_conv(m, gain)

    # Then, override heads with dedicated gains
    for head in policy_heads:
        if isinstance(head, (nn.Linear, nn.Conv2d)):
            _init_linear_or_conv(head, 0.01)
    for head in value_heads:
        if isinstance(head, (nn.Linear, nn.Conv2d)):
            _init_linear_or_conv(head, 1.0)
