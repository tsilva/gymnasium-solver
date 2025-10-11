from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import threading

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn

import wandb
from agents.hyperparameter_mixin import HyperparameterMixin
from utils.config import Config
from utils.decorators import must_implement
from utils.io import write_json
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

    def build_env(self, stage: str, **kwargs):
        from utils.environment import build_env_from_config, _is_alepy_env_id, _is_stable_retro_env_id

        # Ensure _envs is initialized
        self._envs = self._envs if hasattr(self, "_envs") else {}

        # When eval_async is enabled, use much fewer eval envs to reduce CPU contention
        # Use 1/4 of training envs (min 4, max 8) so eval can lag behind without blocking training
        if self.config.eval_async and stage == "val" and "n_envs" not in kwargs:
            eval_n_envs = max(4, min(8, self.config.n_envs // 4))
            kwargs["n_envs"] = eval_n_envs

        reuse_alepy_vectorization = (
            _is_alepy_env_id(self.config.env_id)
            and getattr(self.config, "obs_type", None) == "rgb"
            and self.config.vectorization_mode in ("auto", "alepy")
            and "vectorization_mode" not in kwargs
        )

        default_kwargs = {
            "train": {
                "seed": self.config.seed_train,
            },
            # Record truncated video of first env (requires vectorization_mode='sync', render_mode="rgb_array")
            # When eval_async is enabled, uses fewer envs (set via kwargs above)
            "val": {
                "seed": self.config.seed_val,
                "vectorization_mode": "sync",
                "render_mode": "rgb_array",
                "record_video": False, #self.config.obs_type == "rgb", # TODO: softcode
                "record_video_kwargs": {
                    "video_length": 100,
                },
            },
            # Record truncated video of first env (requires vectorization_mode='sync', render_mode="rgb_array")
            "test": {
                "seed": self.config.seed_test,
                "vectorization_mode": "sync",
                "render_mode": "rgb_array",
                "record_video": False, #self.config.obs_type == "rgb", # TODO: softcode
                "record_video_kwargs": {
                    "video_length": None,
                },
            },
        }

        if reuse_alepy_vectorization:
            default_kwargs["val"]["vectorization_mode"] = self.config.vectorization_mode
            default_kwargs["test"]["vectorization_mode"] = self.config.vectorization_mode

        # stable-retro doesn't support multiple emulator instances per process
        # Force async vectorization for val/test stages and disable video recording
        if _is_stable_retro_env_id(self.config.env_id) and stage in ("val", "test"):
            kwargs.setdefault("n_envs", 1)
            kwargs["vectorization_mode"] = "async"
            kwargs["record_video"] = False  # async mode doesn't support video recording

        # Build the environment
        self._envs[stage] = build_env_from_config(
            self.config, **{
                **default_kwargs[stage],
                **kwargs,
            }
        )
            
    def get_env(self, stage: str):
        return self._envs[stage]

    def build_rollout_collector(self, stage: str):
        from utils.rollouts import RolloutCollector

        # Ensure _rollout_collectors is initialized
        self._rollout_collectors = self._rollout_collectors if hasattr(self, "_rollout_collectors") else {}

        # Build the rollout collector
        self._rollout_collectors[stage] = RolloutCollector(
            self.get_env(stage),
            self.policy_model,
            n_steps=self.config.n_steps,
            **{
                **self.config.rollout_collector_hyperparams(),
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,      
                "returns_type": self.config.returns_type,
                "normalize_returns": self.config.normalize_returns == "rollout",
                "advantages_type": self.config.advantages_type,
                "normalize_advantages": self.config.normalize_advantages == "rollout",
            }
        )

    def attach_print_metrics_logger(self, logger: "MetricsTableLogger") -> None:
        self._print_metrics_logger = logger

    def _set_stage_display(self, stage: str) -> None:
        if self._print_metrics_logger is None:
            return
        self._print_metrics_logger.set_stage(stage)

    def get_rollout_collector(self, stage: str):
        return self._rollout_collectors[stage]

    def on_fit_start(self):
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
            current_env_steps = train_metrics.get("cnt/total_env_steps", 0)
            next_rollout_steps = self.config.n_envs * self.config.n_steps
            would_exceed = (current_env_steps + next_rollout_steps) > self.config.max_env_steps

            if would_exceed:
                from utils.formatting import format_metric_value
                current_s = format_metric_value("train/cnt/total_env_steps", current_env_steps)
                limit_s = format_metric_value("train/cnt/total_env_steps", self.config.max_env_steps)
                reason = f"'train/cnt/total_env_steps': {current_s} + {next_rollout_steps} would exceed {limit_s}."
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
        """Launch async evaluation in background thread.

        If eval is already running, marks current epoch as pending for re-evaluation
        instead of blocking. When the running eval completes, it will automatically
        launch evaluation for the pending epoch.

        Args:
            eval_epoch: Epoch to evaluate. If None, uses current_epoch.
        """
        if self._async_eval_shutdown.is_set():
            return
        if eval_epoch is None:
            eval_epoch = int(self.current_epoch)

        # If eval is already running, mark this epoch as pending and return
        if self._async_eval_thread is not None and self._async_eval_thread.is_alive():
            if self._async_eval_shutdown.is_set():
                return
            with self._async_eval_lock:
                self._async_eval_pending_epoch = eval_epoch
            return

        # Mark which epoch we're evaluating
        with self._async_eval_lock:
            self._async_eval_running_epoch = eval_epoch
            self._async_eval_pending_epoch = None

        def _run_eval():
            # Capture the epoch we're evaluating (thread-safe)
            with self._async_eval_lock:
                current_eval_epoch = self._async_eval_running_epoch

            if self._async_eval_shutdown.is_set():
                return

            val_collector = self.get_rollout_collector("val")
            val_metrics = val_collector.evaluate_episodes(
                n_episodes=self.config.eval_episodes,
                deterministic=self.config.eval_deterministic,
            )

            # Compute FPS metrics
            epoch_fps_values = self.timings.throughput_since("on_validation_epoch_start", values_now=val_metrics)
            epoch_fps = epoch_fps_values.get("cnt/total_vec_steps", epoch_fps_values.get("roll/vec_steps", 0.0))

            # Store results in shared dict with lock, including which epoch was evaluated
            with self._async_eval_lock:
                self._async_eval_metrics = {
                    **val_metrics,
                    "cnt/epoch": int(current_eval_epoch),
                    "eval/model_epoch": int(current_eval_epoch),  # Track which model was evaluated
                    "epoch_fps": epoch_fps,
                }
                self._async_eval_running_epoch = None
                pending_epoch = self._async_eval_pending_epoch
                # Clear pending now that we've captured it
                self._async_eval_pending_epoch = None

            # Trigger early stopping check if trainer is available
            if hasattr(self, 'trainer') and self.trainer is not None:
                # Run callbacks on_validation_epoch_end to check early stopping
                # This is safe because callbacks are designed to be called from any thread
                for callback in self.trainer.callbacks:
                    if hasattr(callback, '_maybe_stop'):
                        callback._maybe_stop(self.trainer, self)

            # If there's a pending epoch, launch eval for it immediately
            if pending_epoch is not None and not self._async_eval_shutdown.is_set():
                self._launch_async_eval(eval_epoch=pending_epoch)

        self._async_eval_thread = threading.Thread(target=_run_eval, daemon=True)
        self._async_eval_thread.start()

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

    def on_fit_end(self):
        # Wait for async eval thread to complete if still running
        self._async_eval_shutdown.set()
        with self._async_eval_lock:
            self._async_eval_pending_epoch = None
        if self._async_eval_thread is not None and self._async_eval_thread.is_alive():
            self._async_eval_thread.join()
        self._async_eval_thread = None

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
        # TODO: create method that encapsulates this logic
        optimizers = self.optimizers()
        if not isinstance(losses, (list, tuple)): losses = [losses]
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        models = [self.policy_model] # TODO: temporary hack

        # Backpropagate and step for each model, loss, and optimizer combo
        for model, loss, optimizer in zip(models, losses, optimizers):
            # Backpropagate loss, zeroing out gradients first 
            # so they're not accumulated from previous steps
            optimizer.zero_grad()
            self.manual_backward(loss) # TODO: this or loss.backward()?
            
            # Compute model gradient norms and log them
            # (do this before any gradient clipping)
            metrics = model.compute_grad_norms()
            self.metrics_recorder.record("train", metrics)

            # Clip gradients using Lightning's built-in method
            # (respects strategy/precision, eg: mixed precision training)
            if self.config.max_grad_norm is not None:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=self.config.max_grad_norm,
                    gradient_clip_algorithm="norm",
                )

            # Perform an optimization step 
            # using the computed gradients
            optimizer.step()

    def calc_training_progress(self):
        """Calculate training progress as a fraction of max_env_steps (0.0 to 1.0)."""
        max_env_steps = self.config.max_env_steps
        if max_env_steps is None: return 0.0
        train_collector = self.get_rollout_collector("train")
        # max_env_steps is env_steps, so use total_timesteps for progress calculation
        total_timesteps = train_collector.total_timesteps
        training_progress = max(0.0, min(total_timesteps / max_env_steps, 1.0))
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
        """Save complete training state to directory for resuming.

        Saves:
        - Model state dict
        - Optimizer state dict(s)
        - Training counters (epoch, timesteps)
        - RNG states (torch, numpy, random, cuda)
        - Config
        - Run ID
        - Metrics recorder state (best rewards, etc.)
        """
        import random
        import numpy as np
        import torch
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)

        # Save model state
        model_path = checkpoint_dir / "model.pt"
        torch.save(self.policy_model.state_dict(), model_path)

        # Save optimizer state(s) if trainer is attached (during training)
        try:
            optimizers = self.optimizers()
            if not isinstance(optimizers, (list, tuple)):
                optimizers = [optimizers]

            optimizer_states = [opt.state_dict() for opt in optimizers]
            optimizer_path = checkpoint_dir / "optimizer.pt"
            torch.save(optimizer_states, optimizer_path)
        except RuntimeError:
            # Trainer not attached yet (e.g., checkpoint being saved before training)
            # This is fine, we'll skip optimizer state
            pass

        # Gather training state
        train_collector = self.get_rollout_collector("train")
        train_metrics = train_collector.get_metrics()

        # Get best rewards from collectors
        val_collector = self.get_rollout_collector("val")

        # Serialize config (same as save_to_json)
        from dataclasses import asdict
        config_dict = asdict(self.config)
        config_dict["algo_id"] = self.config.algo_id  # Add algo_id explicitly

        state = {
            "epoch": int(self.current_epoch),
            "total_timesteps": train_metrics.get("cnt/total_env_steps", 0),
            "total_vec_steps": train_metrics.get("cnt/total_vec_steps", 0),
            "run_id": self.run.run_id if self.run else None,
            "config": config_dict,
            "best_train_reward": float(train_collector._best_episode_reward),
            "best_val_reward": float(val_collector._best_episode_reward),
            "rng_states": {
                "torch": torch.get_rng_state().tolist(),
                "torch_cuda": [s.tolist() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
                "numpy": {
                    "state_type": np.random.get_state()[0],
                    "state_keys": np.random.get_state()[1].tolist(),
                    "state_pos": int(np.random.get_state()[2]),
                    "state_has_gauss": int(np.random.get_state()[3]),
                    "state_cached_gaussian": float(np.random.get_state()[4]),
                },
                "random": random.getstate(),
            },
        }

        # Save state as JSON
        state_path = checkpoint_dir / "state.json"
        from utils.io import write_json
        write_json(state_path, state)

    def load_checkpoint(self, checkpoint_dir: Path, resume_training: bool = True, strict: bool = True) -> None:
        """Restore complete training state from directory.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            resume_training: If True, restore optimizer and RNG states for exact resumption
            strict: If True, require exact match of model state dict keys. If False, allow partial loading (useful for transfer learning)
        """
        import random
        import numpy as np
        import torch
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)

        # Load model state (support both new and old formats)
        model_path = checkpoint_dir / "model.pt"
        old_model_path = checkpoint_dir / "policy.ckpt"

        def _load_state_dict_flexible(state_dict, strict):
            """Load state dict with flexible matching for transfer learning."""
            if strict:
                return self.policy_model.load_state_dict(state_dict, strict=True)

            # Filter state dict to only include keys with matching shapes
            model_state = self.policy_model.state_dict()
            filtered_state = {}
            size_mismatches = []

            for key, value in state_dict.items():
                if key in model_state:
                    if value.shape == model_state[key].shape:
                        filtered_state[key] = value
                    else:
                        size_mismatches.append(key)

            result = self.policy_model.load_state_dict(filtered_state, strict=False)
            loaded = len(filtered_state)
            skipped = len(size_mismatches)
            missing = len(result.missing_keys)

            if loaded > 0:
                print(f"Partial weight loading: {loaded} params loaded, {skipped} skipped (size mismatch), {missing} missing")
            else:
                print(f"Warning: No compatible weights found for transfer learning")

            return result

        if model_path.exists():
            model_state = torch.load(model_path, map_location='cpu', weights_only=True)
            _load_state_dict_flexible(model_state, strict)
        elif old_model_path.exists():
            # Old checkpoint format
            print("Loading from old checkpoint format (policy.ckpt)")
            checkpoint = torch.load(old_model_path, map_location='cpu', weights_only=False)
            if "model_state_dict" in checkpoint:
                _load_state_dict_flexible(checkpoint["model_state_dict"], strict)
            else:
                # Even older format, direct state dict
                _load_state_dict_flexible(checkpoint, strict)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path} or {old_model_path}")

        # Load state JSON (if it exists)
        state_path = checkpoint_dir / "state.json"
        if state_path.exists():
            from utils.io import read_json
            state = read_json(state_path)
        else:
            # Old checkpoint without state.json
            print("Warning: Checkpoint missing state.json, skipping optimizer/RNG restoration")
            state = None
            resume_training = False  # Can't fully resume without state

        # If resuming training, restore optimizer and RNG states
        if resume_training and state:
            # Load optimizer state(s) - only works if trainer is attached
            optimizer_path = checkpoint_dir / "optimizer.pt"
            if optimizer_path.exists():
                try:
                    optimizer_states = torch.load(optimizer_path, map_location='cpu', weights_only=False)
                    if not isinstance(optimizer_states, list):
                        optimizer_states = [optimizer_states]

                    optimizers = self.optimizers()
                    if not isinstance(optimizers, (list, tuple)):
                        optimizers = [optimizers]

                    for opt, opt_state in zip(optimizers, optimizer_states):
                        opt.load_state_dict(opt_state)
                except RuntimeError:
                    # Trainer not attached yet, optimizer will be initialized fresh
                    print("Note: Optimizer will be initialized fresh (trainer not attached yet)")
                    pass

            # Restore RNG states
            if state and "rng_states" in state:
                rng_states = state["rng_states"]
                torch.set_rng_state(torch.ByteTensor(rng_states["torch"]))
                if rng_states.get("torch_cuda") and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([torch.ByteTensor(s) for s in rng_states["torch_cuda"]])

                # Restore numpy RNG state
                numpy_state = rng_states["numpy"]
                np_state_tuple = (
                    numpy_state["state_type"],
                    np.array(numpy_state["state_keys"], dtype=np.uint32),
                    numpy_state["state_pos"],
                    numpy_state["state_has_gauss"],
                    numpy_state["state_cached_gaussian"],
                )
                np.random.set_state(np_state_tuple)

                # Restore random module state
                # random.getstate() returns a tuple that we need to reconstruct
                random_state = rng_states["random"]
                # Convert list back to proper tuple structure
                random.setstate((random_state[0], tuple(random_state[1]), random_state[2]))

            # Restore best episode rewards in collectors
            if state and "best_train_reward" in state and state["best_train_reward"] is not None:
                train_collector = self.get_rollout_collector("train")
                train_collector._best_episode_reward = float(state["best_train_reward"])
            if state and "best_val_reward" in state and state["best_val_reward"] is not None:
                val_collector = self.get_rollout_collector("val")
                val_collector._best_episode_reward = float(state["best_val_reward"])

        # Print checkpoint info
        if state:
            epoch = state.get("epoch", "unknown")
            total_timesteps = state.get("total_timesteps", "unknown")
            best_train = state.get("best_train_reward", "unknown")
            best_val = state.get("best_val_reward", "unknown")

            print(f"Checkpoint loaded from epoch {epoch}:")
            print(f"  Total timesteps: {total_timesteps}")
            print(f"  Best train reward: {best_train}")
            print(f"  Best val reward: {best_val}")
        else:
            print("Checkpoint loaded (model weights only, no training state)")
