from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn

import wandb
from utils.config import Config
from utils.decorators import must_implement
from utils.formatting import sanitize_name
from utils.io import write_json
from utils.metric_bundles import CoreMetricAlerts
from utils.metrics_monitor import MetricsMonitor
from utils.metrics_recorder import MetricsRecorder
from utils.rollouts import RolloutCollector, RolloutTrajectory
from utils.run import Run
from utils.timings_tracker import TimingsTracker

STAGES = ["train", "val", "test"]

class BaseAgent(pl.LightningModule):
    
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

    @must_implement
    def build_models(self):
        # Subclasses must implement this to create their own models (eg: policy, value)
        pass

    @must_implement
    def losses_for_batch(self, batch, batch_idx):
        # Subclasses must implement this to compute the losses for each training steps' batch
        pass

    def build_env(self, stage: str, **kwargs):
        from utils.environment import build_env_from_config

        # Ensure _envs is initialized
        self._envs = self._envs if hasattr(self, "_envs") else {}

        default_kwargs = {
            "train": {
                "seed": self.config.seed,
            },
            # Record truncated video of first env (requires vectorization_mode='sync', render_mode="rgb_array")
            "val": {
                "seed": self.config.seed + 1000,
                "vectorization_mode": "sync",
                "render_mode": "rgb_array",
                "record_video": True,
                "record_video_kwargs": {
                    "video_length": 100,
                },
            },
            # Record truncated video of first env (requires vectorization_mode='sync', render_mode="rgb_array")
            "test": {
                "seed": self.config.seed + 2000,
                "vectorization_mode": "sync",
                "render_mode": "rgb_array",
                "record_video": True,
                "record_video_kwargs": {
                    "video_length": None,
                },
            },
        }

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
        # Start epoch timer
        train_collector = self.get_rollout_collector("train")
        train_metrics = train_collector.get_metrics()
        self.timings.start("on_train_epoch_start", values=train_metrics)

        # Read latest hyperparameters from run 
        # (may have been changed by user during training)
        self._read_hyperparameters_from_run()

        # Log hyperparameters that are tunable in real-time
        self._log_hyperparameters()

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
        print("Running validation...")
        val_collector = self.get_rollout_collector("val")
        val_metrics = val_collector.get_metrics()
        self.timings.start("on_validation_epoch_start", values=val_metrics)

    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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
        pass

    def on_fit_end(self):
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
        with test_env.recorder(str(video_path), record_video=True):
            test_collector = self.get_rollout_collector("test")
            final_metrics = test_collector.evaluate_episodes(
                n_episodes=1,
                deterministic=self.config.eval_deterministic,
            )
            json_path = video_path.with_suffix(".json")
            from utils.metrics_serialization import prepare_metrics_for_json
            write_json(json_path, prepare_metrics_for_json(final_metrics))
    
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
        loggers = self._build_trainer_loggers()

        # Build trainer callbacks
        callbacks = self._build_trainer_callbacks()

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
    
    # -------------------------
    # Pre-prompt guidance helpers
    # -------------------------
  
    def _build_trainer_loggers__wandb(self):
        from dataclasses import asdict

        from pytorch_lightning.loggers import WandbLogger

        import wandb

        # Create the wandb logger, attach to the existing run if present
        project_name = self.config.project_id if self.config.project_id else sanitize_name(self.config.env_id)
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=True,
            config=asdict(self.config),
        ) if wandb.run is None else WandbLogger(log_model=True)

        # Define the common step metric
        from utils.metrics_config import metrics_config
        wandb_run = wandb_logger.experiment
        wandb_run.define_metric("*", step_metric=metrics_config.total_timesteps_key())

        # Change the run name to {algo_id}-{run_id}
        wandb_run.name = f"{self.config.algo_id}-{wandb_run.id}"

        # TODO: review if log_freq makes sense, perhaps tie to eval_freq?
        # Log model gradients to wandb
        wandb_logger.watch(self.policy_model, log="gradients", log_freq=100)
        
        return wandb_logger

    def _build_trainer_loggers__csv(self):
        from loggers.metrics_csv_lightning_logger import MetricsCSVLightningLogger
        # TODO: queue_size?
        csv_path = self.run.ensure_metrics_path() # TODO: pass run inside logger?
        csv_logger = MetricsCSVLightningLogger(csv_path=str(csv_path))
        return csv_logger
    
    def _build_trainer_loggers__print(self):
        from loggers.metrics_table_logger import MetricsTableLogger
        print_logger = MetricsTableLogger(metrics_monitor=self.metrics_monitor, run=self.run)
        return print_logger
        
    def _build_trainer_loggers(self):
        # Initialize list of loggers
        loggers = []

        # Prepare a wandb logger (only if enabled)
        if getattr(self.config, 'enable_wandb', True):
            wandb_logger = self._build_trainer_loggers__wandb()
            loggers.append(wandb_logger)

        # Prepare a CSV Lightning logger writing to runs/<id>/metrics.csv
        csv_logger = self._build_trainer_loggers__csv()
        loggers.append(csv_logger)

        # Prepare a terminal print logger that formats metrics from the unified logging stream
        print_logger = self._build_trainer_loggers__print()
        loggers.append(print_logger)

        # Return the loggers
        return loggers

    def _build_trainer_callbacks(self):
        """Assemble trainer callbacks, with an optional end-of-training report."""
        # Lazy imports to avoid heavy deps at module import time
        from trainer_callbacks import (
            ConsoleSummaryCallback,  # TODO; call this something else
            DispatchMetricsCallback,
            EarlyStoppingCallback,
            HyperparameterSchedulerCallback,
            ModelCheckpointCallback,
            MonitorMetricsCallback,
            UploadRunCallback,
            WandbVideoLoggerCallback,
            WarmupEvalCallback,
        )

        # Initialize callbacks list
        callbacks = []
        
        # In case eval warmup is active, add a callback to enable validation only after warmup
        if self.config.eval_warmup_epochs > 0: 
            callbacks.append(WarmupEvalCallback(
                warmup_epochs=self.config.eval_warmup_epochs, 
                eval_freq_epochs=self.config.eval_freq_epochs
            ))

        # Monitor metrics: add alerts if any are triggered
        # (must be before DispatchMetricsCallback because alerts are reported as metrics)
        callbacks.append(MonitorMetricsCallback())

        # Metrics dispatcher: aggregates epoch metrics and logs to Lightning
        callbacks.append(DispatchMetricsCallback())

        # Auto-wire hyperparameter schedulers: scan config for any *_schedule fields
        schedule_suffix = "_schedule"
        max_env_steps = self.config.max_env_steps
        n_envs = self.config.n_envs

        def _schedule_pos_to_vec_steps(raw: Optional[float], *, param: str, default_to_max: bool) -> float:
            """Convert schedule position (fraction or env_steps) to vec_steps.

            Args:
                raw: Schedule position as fraction (0-1) or absolute env_steps (>1)
                param: Parameter name for error messages
                default_to_max: If True and raw is None, default to max_env_steps

            Returns:
                Position in vec_steps
            """
            if raw is None:
                if default_to_max:
                    if max_env_steps is None:
                        raise ValueError(
                            f"{param}_schedule requires config.max_env_steps or an explicit {param}_schedule_end."
                        )
                    # max_env_steps is in env_steps, convert to vec_steps
                    return float(max_env_steps) / n_envs
                return 0.0

            value = float(raw)
            if value < 0.0:
                raise ValueError(f"{param}_schedule start/end must be non-negative.")

            # Fractional position: interpret as fraction of max_env_steps
            if value <= 1.0:
                if max_env_steps is None:
                    raise ValueError(
                        f"{param}_schedule uses fractional start/end but config.max_env_steps is not set."
                    )
                env_steps = value * float(max_env_steps)
                return env_steps / n_envs

            # Absolute position: interpret as env_steps, convert to vec_steps
            return value / n_envs

        def _set_policy_lr(module: "BaseAgent", lr: float) -> None:
            module._change_optimizers_lr(lr)

        for key, value in vars(self.config).items():  # TODO: config.get_schedules()?
            if not key.endswith(schedule_suffix) or not value:
                continue

            param = key[: -len(schedule_suffix)]
            assert hasattr(self, param), f"Module {self} has no attribute {param}"

            start_value = getattr(self.config, f"{param}_schedule_start_value", None)
            end_value = getattr(self.config, f"{param}_schedule_end_value", None)
            if start_value is None or end_value is None:
                raise ValueError(f"{param}_schedule requires start/end values in the config.")

            start_pos_raw = getattr(self.config, f"{param}_schedule_start", None)
            end_pos_raw = getattr(self.config, f"{param}_schedule_end", None)

            start_step = _schedule_pos_to_vec_steps(start_pos_raw, param=param, default_to_max=False)
            end_step = _schedule_pos_to_vec_steps(end_pos_raw, param=param, default_to_max=True)
            warmup_fraction = getattr(self.config, f"{param}_schedule_warmup", 0.0)

            callbacks.append(
                HyperparameterSchedulerCallback(
                    schedule=value,
                    parameter=param,
                    start_value=float(start_value),
                    end_value=float(end_value),
                    start_step=float(start_step),
                    end_step=float(end_step),
                    warmup_fraction=float(warmup_fraction),
                    set_value_fn={"policy_lr": _set_policy_lr}.get(param, None),
                )
            )

        # Early stopping callbacks: must run BEFORE ModelCheckpointCallback
        # so that trainer.should_stop is set when checkpoint logic runs

        # If defined in config, early stop after reaching a certain number of environment steps
        if self.config.max_env_steps: callbacks.append(
            EarlyStoppingCallback("train/cnt/total_env_steps", self.config.max_env_steps)
        )

        # If defined in config, early stop when mean training reward reaches a threshold
        # Config value can be True (use env spec threshold) or a float (override threshold)
        if self.config.early_stop_on_train_threshold:
            train_env = self.get_env("train")
            if isinstance(self.config.early_stop_on_train_threshold, float):
                reward_threshold = self.config.early_stop_on_train_threshold
            else:
                reward_threshold = train_env.get_return_threshold()
            callbacks.append(EarlyStoppingCallback("train/roll/ep_rew/mean", reward_threshold))

        # If defined in config, early stop when mean validation reward reaches a threshold
        # Config value can be True (use env spec threshold) or a float (override threshold)
        if self.config.early_stop_on_eval_threshold:
            val_env = self.get_env("val")
            if isinstance(self.config.early_stop_on_eval_threshold, float):
                reward_threshold = self.config.early_stop_on_eval_threshold
            else:
                reward_threshold = val_env.get_return_threshold()
            callbacks.append(EarlyStoppingCallback("val/roll/ep_rew/mean", reward_threshold))

        # Checkpointing: save best/last models and metrics
        # Must run AFTER early stopping callbacks to detect trainer.should_stop
        callbacks.append(ModelCheckpointCallback( # TODO: pass run
            run=self.run,
            metric="val/roll/ep_rew/mean",
            mode="max"
        ))

        # Video logger watches checkpoints directory recursively for videos (only if W&B is enabled)
        if getattr(self.config, 'enable_wandb', True):
            callbacks.append(WandbVideoLoggerCallback(
                media_root=self.run.checkpoints_dir,
                namespace_depth=1,
            ))

        # Also print a terminal summary and alerts recap at the end of training
        callbacks.append(ConsoleSummaryCallback())

        # Optionally upload run folder to W&B after training completes
        if getattr(self.config, 'enable_wandb', True):
            callbacks.append(UploadRunCallback(run_dir=self.run.run_dir))

        return callbacks

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

    # TODO: is there an util for this?
    def _change_optimizers_lr(self, lr):
        # Keep attribute in sync for logging/inspection
        self.policy_lr = lr
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = lr

    # TODO: review this method
    def configure_optimizers(self):
        from utils.optimizer_factory import build_optimizer
        return build_optimizer(
            params=self.policy_model.parameters(),
            optimizer=self.config.optimizer,
            lr=self.policy_lr, # TODO: is this taking annealing into account?
        )
    
    def _read_hyperparameters_from_run(self):
        from dataclasses import asdict
        loaded_config = asdict(self.run.load_config())
        current_config = asdict(self.config)

        # Identify parameters with active schedules (to skip reloading them)
        scheduled_params = set()
        for key in current_config.keys():
            if key.endswith("_schedule") and current_config.get(key):
                param = key[: -len("_schedule")]
                scheduled_params.add(param)

        changes_map = {}
        for key, value in loaded_config.items():
            if type(value) in [list, tuple, dict, None]: continue
            # Skip parameters with active schedules
            if key in scheduled_params: continue
            current_value = current_config.get(key, None)
            if value != current_value: changes_map[key] = value

        if changes_map: self.on_hyperparams_change(changes_map)
        
    def on_hyperparams_change(self, changes_map):
        for key, value in changes_map.items():
            if not hasattr(self.config, key): continue
            setattr(self.config, key, value)
            if key == "policy_lr": self._change_optimizers_lr(value)
            elif key == "clip_range": self.clip_range = value
            elif key == "vf_coef": self.vf_coef = value
            elif key == "ent_coef": self.ent_coef = value
            elif key == "n_epochs": self._change_n_epochs(value)
        print(f"Hyperparameters changed from run: {changes_map}")

    def _change_n_epochs(self, n_epochs):
        self.n_epochs = n_epochs
        self._train_dataloader.sampler.num_passes = n_epochs

    def _log_hyperparameters(self):
        metrics = {
            "n_epochs": self.n_epochs,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "clip_range": self.clip_range,
            "policy_lr": self.policy_lr,
        }
        prefixed = {f"hp/{k}": v for k, v in metrics.items()}
        self.metrics_recorder.record("train", prefixed)

    # -------------------------
    # Public API for callbacks
    # -------------------------

    def set_early_stop_reason(self, reason: str) -> None:
        """Set the early stopping reason. Called by EarlyStoppingCallback."""
        self._early_stop_reason = reason

    def set_hyperparameter(self, param: str, value: float) -> None:
        """Set a hyperparameter value. Called by HyperparameterSchedulerCallback."""
        setattr(self, param, value)
        if hasattr(self.config, param):
            setattr(self.config, param, value)

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

    def load_checkpoint(self, checkpoint_dir: Path, resume_training: bool = True) -> None:
        """Restore complete training state from directory.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            resume_training: If True, restore optimizer and RNG states for exact resumption
        """
        import random
        import numpy as np
        import torch
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_dir)

        # Load model state (support both new and old formats)
        model_path = checkpoint_dir / "model.pt"
        old_model_path = checkpoint_dir / "policy.ckpt"

        if model_path.exists():
            model_state = torch.load(model_path, map_location='cpu', weights_only=True)
            self.policy_model.load_state_dict(model_state)
        elif old_model_path.exists():
            # Old checkpoint format
            print("Loading from old checkpoint format (policy.ckpt)")
            checkpoint = torch.load(old_model_path, map_location='cpu', weights_only=False)
            if "model_state_dict" in checkpoint:
                self.policy_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Even older format, direct state dict
                self.policy_model.load_state_dict(checkpoint)
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
