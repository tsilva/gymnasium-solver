"""BaseAgent helpers for asynchronous validation evaluation."""

from __future__ import annotations

import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


def launch_async_eval(agent: "BaseAgent", eval_epoch: Optional[int] = None) -> None:
    """Launch asynchronous validation without blocking the training loop."""
    if agent._async_eval_shutdown.is_set():
        return
    if eval_epoch is None:
        eval_epoch = int(agent.current_epoch)

    if agent._async_eval_thread is not None and agent._async_eval_thread.is_alive():
        if agent._async_eval_shutdown.is_set():
            return
        with agent._async_eval_lock:
            agent._async_eval_pending_epoch = eval_epoch
        return

    with agent._async_eval_lock:
        agent._async_eval_running_epoch = eval_epoch
        agent._async_eval_pending_epoch = None

    if hasattr(agent, "_eval_models") and "val" in agent._eval_models:
        agent._eval_models["val"].load_state_dict(agent.policy_model.state_dict())

    def _run_eval() -> None:
        with agent._async_eval_lock:
            current_eval_epoch = agent._async_eval_running_epoch

        if agent._async_eval_shutdown.is_set():
            return

        val_collector = agent.get_rollout_collector("val")
        val_metrics = val_collector.evaluate_episodes(
            n_episodes=agent.config.eval_episodes,
            deterministic=agent.config.eval_deterministic,
        )

        epoch_fps_values = agent.timings.throughput_since(
            "on_validation_epoch_start",
            values_now=val_metrics,
        )
        epoch_fps = epoch_fps_values.get(
            "cnt/total_vec_steps",
            epoch_fps_values.get("roll/vec_steps", 0.0),
        )

        with agent._async_eval_lock:
            agent._async_eval_metrics = {
                **val_metrics,
                "cnt/epoch": int(current_eval_epoch),
                "eval/model_epoch": int(current_eval_epoch),
                "epoch_fps": epoch_fps,
            }
            agent._async_eval_running_epoch = None
            pending_epoch = agent._async_eval_pending_epoch
            agent._async_eval_pending_epoch = None

        if hasattr(agent, "trainer") and agent.trainer is not None:
            for callback in agent.trainer.callbacks:
                if hasattr(callback, "_maybe_stop"):
                    callback._maybe_stop(agent.trainer, agent)

        if pending_epoch is not None and not agent._async_eval_shutdown.is_set():
            launch_async_eval(agent, eval_epoch=pending_epoch)

    agent._async_eval_thread = threading.Thread(target=_run_eval, daemon=True)
    agent._async_eval_thread.start()


def cleanup_async_eval(agent: "BaseAgent") -> None:
    """Join any outstanding async validation work and clear pending state."""
    if not agent.config.eval_async:
        return

    agent._async_eval_shutdown.set()
    with agent._async_eval_lock:
        agent._async_eval_pending_epoch = None
    if agent._async_eval_thread is not None and agent._async_eval_thread.is_alive():
        agent._async_eval_thread.join(timeout=5.0)
    agent._async_eval_thread = None
