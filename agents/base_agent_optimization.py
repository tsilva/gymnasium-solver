"""BaseAgent helpers for manual optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


def backpropagate_and_step(agent: "BaseAgent", losses) -> None:
    """Backpropagate and apply one optimizer step per configured model/loss."""
    optimizers = agent.optimizers()
    if not isinstance(losses, (list, tuple)):
        losses = [losses]
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    models = [agent.policy_model]

    for model, loss, optimizer in zip(models, losses, optimizers):
        optimizer.zero_grad()
        agent.manual_backward(loss)

        metrics = model.compute_grad_norms()
        agent.metrics_recorder.record("train", metrics)

        if agent.config.max_grad_norm is not None:
            agent.clip_gradients(
                optimizer,
                gradient_clip_val=agent.config.max_grad_norm,
                gradient_clip_algorithm="norm",
            )

        optimizer.step()
