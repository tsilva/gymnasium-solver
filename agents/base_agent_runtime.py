"""Runtime orchestration helpers for BaseAgent.

These helpers keep BaseAgent's public Lightning hooks stable while moving the
order-sensitive training lifecycle into a focused module.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import wandb

from agents.base_agent_async_eval import sync_eval_model
from agents.base_agent_checkpointing import restore_deferred_optimizer_states
from utils.run import Run

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


def on_fit_start(agent: "BaseAgent") -> None:
    restore_deferred_optimizer_states(agent)

    train_collector = agent.get_rollout_collector("train")
    train_metrics = train_collector.get_metrics()
    agent.timings.start("on_fit_start", values=train_metrics)


def train_dataloader(agent: "BaseAgent"):
    resume_epoch = getattr(agent, "_resume_from_epoch", None)
    is_resuming = resume_epoch is not None
    assert agent.current_epoch == 0 or is_resuming, (
        "train_dataloader should only be called once at the start of training"
    )

    train_collector = agent.get_rollout_collector("train")
    agent._trajectories = train_collector.collect()

    from utils.dataloaders import build_index_collate_loader_from_collector
    from utils.random import get_global_torch_generator

    generator = get_global_torch_generator(agent.config.seed)
    agent._train_dataloader = build_index_collate_loader_from_collector(
        collector=train_collector,
        trajectories_getter=lambda: agent._trajectories,
        batch_size=agent.config.batch_size,
        num_passes=agent.config.n_epochs,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    return agent._train_dataloader


def on_train_epoch_start(agent: "BaseAgent") -> None:
    agent._early_stop_epoch = False

    if agent.config.eval_async:
        with agent._async_eval_lock:
            agent._async_eval_metrics = {}

    agent._set_stage_display("train")
    train_collector = agent.get_rollout_collector("train")
    train_metrics = train_collector.get_metrics()
    agent.timings.start("on_train_epoch_start", values=train_metrics)

    agent._read_hyperparameters_from_run()
    agent._log_hyperparameters()

    if agent.config.max_env_steps is not None:
        current_env_steps = train_metrics.get("cnt/total_env_steps", 0)
        next_rollout_env_steps = agent.config.n_envs * agent.config.n_steps
        would_exceed = (
            current_env_steps + next_rollout_env_steps
        ) > agent.config.max_env_steps

        if would_exceed:
            from utils.formatting import format_metric_value

            current_s = format_metric_value(
                "train/cnt/total_env_steps",
                current_env_steps,
            )
            limit_s = format_metric_value(
                "train/cnt/total_env_steps",
                agent.config.max_env_steps,
            )
            reason = (
                f"'train/cnt/total_env_steps': {current_s} + "
                f"{next_rollout_env_steps} would exceed {limit_s}."
            )
            print(f"Early stopping! {reason}")
            agent.set_early_stop_reason(reason)
            agent.trainer.should_stop = True
            return

    if int(agent.current_epoch) > 0:
        agent._trajectories = train_collector.collect()


def training_step(agent: "BaseAgent", batch, batch_idx):
    if agent._early_stop_epoch:
        return None

    detailed_metrics = getattr(agent.config, "detailed_optimization_metrics", False)
    agent.policy_model._track_activations = bool(detailed_metrics)
    result = agent.losses_for_batch(batch, batch_idx)

    agent.policy_model._track_activations = False
    if detailed_metrics:
        activation_metrics = agent.policy_model.compute_activation_stats()
        if activation_metrics:
            agent.metrics_recorder.record("train", activation_metrics)

    early_stop_epoch = result["early_stop_epoch"]
    if early_stop_epoch:
        agent._early_stop_epoch = True
        return None

    losses = result["loss"]
    agent._backpropagate_and_step(losses)
    return None


def val_dataloader(agent: "BaseAgent"):
    from utils.dataloaders import build_dummy_loader

    return build_dummy_loader()


def on_validation_epoch_start(agent: "BaseAgent") -> None:
    agent._set_stage_display("val")
    val_collector = agent.get_rollout_collector("val")
    val_metrics = val_collector.get_metrics()
    agent.timings.start("on_validation_epoch_start", values=val_metrics)

    if agent.config.eval_async:
        agent._launch_async_eval()


def validation_step(agent: "BaseAgent", batch, batch_idx, dataloader_idx=0):
    del batch, batch_idx, dataloader_idx

    if agent.config.eval_async:
        return

    sync_eval_model(agent, stage="val")

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
    agent.metrics_recorder.record(
        "val",
        {
            **val_metrics,
            "cnt/epoch": int(agent.current_epoch),
            "epoch_fps": epoch_fps,
        },
    )


def on_validation_epoch_end(agent: "BaseAgent") -> None:
    if agent.config.eval_async:
        with agent._async_eval_lock:
            if agent._async_eval_metrics:
                agent.metrics_recorder.record("val", agent._async_eval_metrics)


def on_fit_end(agent: "BaseAgent") -> None:
    agent._cleanup_async_eval()

    if getattr(agent, "_aborted_before_training", False):
        agent._final_stop_reason = getattr(
            agent,
            "_early_stop_reason",
            "User aborted before training.",
        )
        return

    time_elapsed = agent.timings.seconds_since("on_fit_start")
    agent._fit_elapsed_seconds = float(time_elapsed)
    agent._final_stop_reason = agent._early_stop_reason

    _test_env = agent.get_env("test")
    video_path = agent.run.checkpoints_dir / "final.mp4"
    del _test_env, video_path


def learn(agent: "BaseAgent") -> None:
    from utils.logging import stream_output_to_log

    if agent.run is None:
        run_id = (
            wandb.run.id
            if wandb.run is not None
            else f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        agent.run = Run.create(
            run_id=run_id,
            config=agent.config,
        )

    log_path = agent.run._ensure_path("run.log")
    with stream_output_to_log(log_path):
        agent._learn()


def _learn(agent: "BaseAgent") -> None:
    from utils.callback_builder import CallbackBuilder
    from utils.trainer_factory import build_trainer
    from utils.trainer_loggers import TrainerLoggersBuilder

    loggers = TrainerLoggersBuilder(agent).build()
    callbacks = CallbackBuilder(agent).build()

    trainer = build_trainer(
        config=agent.config,
        logger=loggers,
        callbacks=callbacks,
    )

    resume_epoch = getattr(agent, "_resume_from_epoch", None)
    if resume_epoch is not None:
        trainer.fit_loop.epoch_progress.current.completed = resume_epoch
        trainer.fit_loop.epoch_progress.current.processed = resume_epoch

    trainer.fit(agent)
