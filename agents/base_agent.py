from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import threading

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn

import wandb
from agents.base_agent_async_eval import cleanup_async_eval, launch_async_eval
from agents.base_agent_bootstrap import build_stage_env, build_stage_rollout_collector
from agents.base_agent_checkpointing import (
    load_agent_checkpoint,
    restore_deferred_optimizer_states,
    save_agent_checkpoint,
)
from agents.base_agent_optimization import backpropagate_and_step
from agents.hyperparameter_mixin import HyperparameterMixin
from utils.config import Config
from utils.decorators import must_implement
from utils.metric_bundles import CoreMetricAlerts
from utils.metrics_monitor import MetricsMonitor
from utils.metrics_recorder import MetricsRecorder
from utils.rollouts import RolloutCollector, RolloutTrajectory
from utils.run import Run
from utils.timings_tracker import TimingsTracker

if TYPE_CHECKING:
    from loggers.metrics_table_logger import MetricsTableLogger

STAGES = ["train", "val", "test"]

class BaseAgent(HyperparameterMixin, pl.LightningModule):
    
    run: Run
    config: Config
    policy_model: nn.Module
    _envs: Dict[str, Any]
    _rollout_collectors: Dict[str, RolloutCollector]
    _trajectories: List[RolloutTrajectory]
    _early_stop_epoch: bool
    _fit_elapsed_seconds: float
    _final_stop_reason: str
    _train_dataloader: DataLoader
    _async_eval_thread: Optional[threading.Thread]
    _async_eval_metrics: Dict[str, Any]
    _async_eval_lock: threading.Lock

    def __init__(self, config):
        super().__init__()

        # Define which hyperparameters are saved along 
        # with module checkpoints (default to all kwargs)
        self.save_hyperparameters()

        # Disable automatic optimization to allow manual optimization, namely for tuning 
        # multiple models with different optimizers (eg: independent policy and value models)
        self.automatic_optimization = False

        # Initialize instance attributes
        self.config = config
        self.run = None
        self.policy_model = None
        self._envs = {}
        self._rollout_collectors = {}
        self._trajectories = []
        self._early_stop_epoch = False
        self._fit_elapsed_seconds = 0.0
        self._final_stop_reason = ""
        self._early_stop_reason = ""
        self._print_metrics_logger: Optional["MetricsTableLogger"] = None
        self._async_eval_thread = None
        self._async_eval_metrics = {}
        self._async_eval_lock = threading.Lock()
        self._async_eval_shutdown = threading.Event()
        self._async_eval_pending_epoch = None  # Tracks if we need to eval a newer model
        self._async_eval_running_epoch = None  # Tracks which epoch is currently being evaluated

        # Initialize schedulable hyperparameters (mutable during training)
        self.policy_lr = config.policy_lr
        self.clip_range = config.clip_range
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.n_epochs = config.n_epochs
        
        # Initialize timing tracker for training 
        # loop performance measurements
        self.timings = TimingsTracker()

        # Metrics recorder aggregates per-epoch metrics (train/eval) and maintains
        # a step-aware numeric history for terminal summaries.
        self.metrics_recorder = MetricsRecorder()

        # Create metrics monitor registry (eg: used for metric alerts)
        self.metrics_monitor = MetricsMonitor(self.metrics_recorder)

        # Register bundle of metric alerts that apply to all algorithms
        core_metric_alerts = CoreMetricAlerts(self)
        self.metrics_monitor.register_bundle(core_metric_alerts)

        # Build the environments
        for stage in STAGES: self.build_env(stage)

        # Build the models (requires environments for shape inference)
        self.build_models()

        # Build the rollout collectors (requires models and environments)
        for stage in STAGES: self.build_rollout_collector(stage)

    def build_models(self):
        from utils.policy_factory import build_policy_from_env_and_config
        train_env = self.get_env("train")
        self.policy_model = build_policy_from_env_and_config(train_env, self.config)

    @must_implement
    def losses_for_batch(self, batch, batch_idx):
        # Subclasses must implement this to compute the losses for each training steps' batch
        pass

    def _normalize_advantages(self, advantages):
        """Normalize advantages if configured, returning normalized tensor and metrics."""
        from utils.torch import normalize_batch_with_metrics
        return normalize_batch_with_metrics(
            advantages, self.config.normalize_advantages, "roll/adv"
        )

    def _build_common_metrics(self, loss, policy_loss, entropy_loss, entropy):
        """Build common metrics shared across algorithms."""
        return {
            'opt/loss/total': loss.detach(),
            'opt/loss/policy': policy_loss.detach(),
            'opt/loss/entropy': entropy_loss.detach(),
            'opt/policy/entropy': entropy.detach(),
        }

    def build_env(self, stage: str, **kwargs):
        build_stage_env(self, stage, **kwargs)
            
    def get_env(self, stage: str):
        return self._envs[stage]

    def build_rollout_collector(self, stage: str):
        build_stage_rollout_collector(self, stage)

    def attach_print_metrics_logger(self, logger: "MetricsTableLogger") -> None:
        self._print_metrics_logger = logger

    def _set_stage_display(self, stage: str) -> None:
        if self._print_metrics_logger is None:
            return
        self._print_metrics_logger.set_stage(stage)

    def get_rollout_collector(self, stage: str):
        return self._rollout_collectors[stage]

    def on_fit_start(self):
        restore_deferred_optimizer_states(self)

        # Start the timing tracker for the entire training run
        train_collector = self.get_rollout_collector("train")
        train_metrics = train_collector.get_metrics()
        self.timings.start("on_fit_start", values=train_metrics)

    def train_dataloader(self):
        # Some lightweight Trainer stubs used in tests don't manage current_epoch on the module.
        # Guard the assertion to avoid AttributeError while still catching repeated calls.
        # When resuming training, current_epoch will be non-zero, so check for _resume_from_epoch
        resume_epoch = getattr(self, '_resume_from_epoch', None)
        is_resuming = resume_epoch is not None
        assert self.current_epoch == 0 or is_resuming, "train_dataloader should only be called once at the start of training"

        # Collect the first rollout
        train_collector = self.get_rollout_collector("train")
        self._trajectories = train_collector.collect()

        # Build efficient index-collate dataloader backed by 
        # MultiPassRandomSampler (allows showing same data N times 
        # per epoch without suffering lightning's epoch turnover costs)
        # TODO: don't use inline imports unless it really makes a big difference, scan entire codebase for this
        from utils.dataloaders import build_index_collate_loader_from_collector
        from utils.random import get_global_torch_generator
        generator = get_global_torch_generator(self.config.seed)
        self._train_dataloader = build_index_collate_loader_from_collector(
            collector=train_collector,
            trajectories_getter=lambda: self._trajectories,
            batch_size=self.config.batch_size,
            num_passes=self.config.n_epochs, # TODO: must allow
            generator=generator,
            # TODO: add support for n_workers and memory options in config if needed
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        return self._train_dataloader
    def on_train_epoch_start(self):
        # Clear previous async eval metrics at start of new train epoch
        # (after early stopping callback has had a chance to check them)
        if self.config.eval_async:
            with self._async_eval_lock:
                self._async_eval_metrics = {}

        self._set_stage_display("train")
        # Start epoch timer
        train_collector = self.get_rollout_collector("train")
        train_metrics = train_collector.get_metrics()
        self.timings.start("on_train_epoch_start", values=train_metrics)

        # Read latest hyperparameters from run
        # (may have been changed by user during training)
        self._read_hyperparameters_from_run()

        # Log hyperparameters that are tunable in real-time
        self._log_hyperparameters()

        # Check if collecting another rollout would exceed max_env_steps budget
        # If so, stop training before the rollout (prevents overshooting the budget)
        if self.config.max_env_steps is not None:
            # train_metrics["cnt/total_env_steps"] is actually total_steps (n_envs * vec_steps)
            current_env_steps = train_metrics.get("cnt/total_env_steps", 0)
            next_rollout_env_steps = self.config.n_envs * self.config.n_steps
            would_exceed = (current_env_steps + next_rollout_env_steps) > self.config.max_env_steps

            if would_exceed:
                from utils.formatting import format_metric_value
                current_s = format_metric_value("train/cnt/total_env_steps", current_env_steps)
                limit_s = format_metric_value("train/cnt/total_env_steps", self.config.max_env_steps)
                reason = f"'train/cnt/total_env_steps': {current_s} + {next_rollout_env_steps} would exceed {limit_s}."
                print(f"Early stopping! {reason}")
                self.set_early_stop_reason(reason)
                self.trainer.should_stop = True
                return

        # Collect fresh trajectories at the start of each training epoch
        # Avoid double-collect on the first epoch: train_dataloader() already
        # collected an initial rollout to bootstrap the dataloader. From epoch 1
        # onward, collect once per epoch to ensure constant timestep growth.
        if int(self.current_epoch) > 0:
            train_collector = self.get_rollout_collector("train")
            self._trajectories = train_collector.collect()

    def training_step(self, batch, batch_idx):
        # In case an early stop was triggered (eg: KL divergence exceeded target), skip batch
        if self._early_stop_epoch:
            return None

        # Enable activation tracking for this forward pass
        self.policy_model._track_activations = True

        # Calculate batch losses
        result = self.losses_for_batch(batch, batch_idx)

        # Compute and log activation statistics
        activation_metrics = self.policy_model.compute_activation_stats()
        if activation_metrics:
            self.metrics_recorder.record("train", activation_metrics)

        # Disable activation tracking
        self.policy_model._track_activations = False

        # TODO: are we sure this stops training on this rollout? remember how we are training multiple epochs on the same rollout
        # In case an early stop was triggered (eg: KL divergence exceeded
        # target, then don't train any more on this rollout, collect a new one)
        early_stop_epoch = result["early_stop_epoch"]
        if early_stop_epoch:
            self._early_stop_epoch = True
            return None

        # Backpropagate losses and update model
        # parameters according to computed gradients
        losses = result["loss"]
        self._backpropagate_and_step(losses)

        # We purposely return None here to avoid
        # triggering Lightning's default optimization logic
        # which would interfere with our manual optimization process
        # (we may need to train multiple models with different optimizers))
        return None

    def on_train_epoch_end(self):
        # Scheduling moved to HyperparameterScheduler callback
        pass
        
    def val_dataloader(self):
        # TODO: should I just do rollouts here?
        from utils.dataloaders import build_dummy_loader
        return build_dummy_loader()

    def on_validation_epoch_start(self):
        self._set_stage_display("val")
        val_collector = self.get_rollout_collector("val")
        val_metrics = val_collector.get_metrics()
        self.timings.start("on_validation_epoch_start", values=val_metrics)

        # If async eval is enabled, launch evaluation in background thread
        if self.config.eval_async:
            self._launch_async_eval()

    def _launch_async_eval(self, eval_epoch: Optional[int] = None):
        launch_async_eval(self, eval_epoch=eval_epoch)

    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # In async mode, validation_step is a no-op (eval runs in background)
        if self.config.eval_async:
            return

        # Run evaluation without recording (videos logged from checkpoints by WandbVideoLoggerCallback)
        val_collector = self.get_rollout_collector("val")
        val_metrics = val_collector.evaluate_episodes(
            n_episodes=self.config.eval_episodes,
            deterministic=self.config.eval_deterministic,
        )

        # Log eval metrics
        epoch_fps_values = self.timings.throughput_since("on_validation_epoch_start", values_now=val_metrics)
        epoch_fps = epoch_fps_values.get("cnt/total_vec_steps", epoch_fps_values.get("roll/vec_steps", 0.0))
        self.metrics_recorder.record("val", {
            **val_metrics,
            "cnt/epoch": int(self.current_epoch),
            "epoch_fps": epoch_fps,
        })

    def on_validation_epoch_end(self):
        # In async mode, check if eval has completed and record metrics if available
        if self.config.eval_async:
            with self._async_eval_lock:
                if self._async_eval_metrics:
                    self.metrics_recorder.record("val", self._async_eval_metrics)
                    # Don't clear metrics yet - early stopping callback needs them
        pass

    def _cleanup_async_eval(self):
        cleanup_async_eval(self)

    def on_exception(self, trainer, pl_module, exception):
        """Handle cleanup when training fails with an exception."""
        self._cleanup_async_eval()

    def on_fit_end(self):
        # Wait for async eval thread to complete if still running
        self._cleanup_async_eval()

        # If user aborted before training, skip finalization work
        if getattr(self, "_aborted_before_training", False):
            self._final_stop_reason = getattr(self, "_early_stop_reason", "User aborted before training.")
            return

        # Persist final duration and stop reason for external reporting
        time_elapsed = self.timings.seconds_since("on_fit_start")
        self._fit_elapsed_seconds = float(time_elapsed)
        self._final_stop_reason = self._early_stop_reason

        # TODO; consider pros/cons of testing vs ensuring a final val when training finishes (less code to maintain)
        # Record final evaluation video and save associated metrics JSON next to it
        test_env = self.get_env("test")
        video_path = self.run.checkpoints_dir / "final.mp4"
        # TODO: restore recording
        #with test_env.recorder(str(video_path), record_video=True):
        #    test_collector = self.get_rollout_collector("test")
        #    final_metrics = test_collector.evaluate_episodes(
        #        n_episodes=1,
        #        deterministic=self.config.eval_deterministic,
        #    )
        #     json_path = video_path.with_suffix(".json")
        #    from utils.metrics_serialization import prepare_metrics_for_json
        #     write_json(json_path, prepare_metrics_for_json(final_metrics))
    
    def learn(self):
        from datetime import datetime
        from utils.logging import stream_output_to_log

        # If run is already attached (resume mode), skip run creation
        if self.run is None:
            # Initialize run directory management and convenience Run accessor
            # Initialize run directory (creates runs/<id>/, checkpoints/, and @last symlink)
            # Generate local run ID when W&B is disabled
            run_id = wandb.run.id if wandb.run is not None else f"local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.run = Run.create(
                run_id=run_id,
                config=self.config
            )

        # TODO: create run context (with run, do log handling inside)
        # Set up comprehensive logging using run-specific logs directory
        log_path = self.run._ensure_path("run.log")
        with stream_output_to_log(log_path): self._learn()

    def _learn(self):
        # Build trainer loggers
        from utils.trainer_loggers import TrainerLoggersBuilder
        loggers = TrainerLoggersBuilder(self).build()

        # Build trainer callbacks
        from utils.callback_builder import CallbackBuilder
        callbacks = CallbackBuilder(self).build()

        # Build the trainer
        from utils.trainer_factory import build_trainer
        trainer = build_trainer(
            config=self.config,
            logger=loggers,
            callbacks=callbacks
        )

        # If resuming from a checkpoint, set the starting epoch
        # (Lightning will increment from this value)
        resume_epoch = getattr(self, '_resume_from_epoch', None)
        if resume_epoch is not None:
            # Set Lightning's internal epoch counter
            trainer.fit_loop.epoch_progress.current.completed = resume_epoch
            trainer.fit_loop.epoch_progress.current.processed = resume_epoch

        # Train the agent
        trainer.fit(self)


    def _backpropagate_and_step(self, losses):
        backpropagate_and_step(self, losses)

    def calc_training_progress(self):
        """Calculate training progress as a fraction of max_env_steps (0.0 to 1.0)."""
        max_env_steps = self.config.max_env_steps
        if max_env_steps is None: return 0.0
        train_collector = self.get_rollout_collector("train")
        # max_env_steps is env_steps, so use total_steps for progress calculation
        total_env_steps = train_collector.total_steps
        training_progress = max(0.0, min(total_env_steps / max_env_steps, 1.0))
        return training_progress

    def configure_optimizers(self):
        from utils.optimizer_factory import build_optimizer
        return build_optimizer(
            params=self.policy_model.parameters(),
            optimizer=self.config.optimizer,
            lr=self.policy_lr, # TODO: is this taking annealing into account?
        )

    # -------------------------
    # Public API for callbacks
    # -------------------------

    def set_early_stop_reason(self, reason: str) -> None:
        """Set the early stopping reason. Called by EarlyStoppingCallback."""
        self._early_stop_reason = reason

    def get_async_eval_metric(self, metric_key: str) -> Optional[float]:
        """Get a metric from async eval results. Returns None if not available."""
        with self._async_eval_lock:
            return self._async_eval_metrics.get(metric_key)

    # -------------------------
    # Checkpoint save/load
    # -------------------------

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        save_agent_checkpoint(self, checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: Path, resume_training: bool = True, strict: bool = True, load_optimizer_only: bool = False) -> None:
        load_agent_checkpoint(
            self,
            checkpoint_dir,
            resume_training=resume_training,
            strict=strict,
            load_optimizer_only=load_optimizer_only,
        )
