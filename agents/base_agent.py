from typing import Any, Dict, List

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

        # Initialize schedulable hyperparameters (mutable during training)
        self.policy_lr = config.policy_lr
        self.clip_range = config.clip_range
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        
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
            # Record truncated video of first env (requires subproc=False, render_mode="rgb_array")
            "val": {
                "seed": self.config.seed + 1000,
                "subproc": False,
                "render_mode": "rgb_array",
                "record_video": True,
                "record_video_kwargs": {
                    "video_length": 100,
                },
            },
            # Record truncated video of first env (requires subproc=False, render_mode="rgb_array")
            "test": {
                "seed": self.config.seed + 2000,
                "subproc": False,
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
        assert self.current_epoch == 0, "train_dataloader should only be called once at the start of training"

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
        return build_index_collate_loader_from_collector(
            collector=train_collector,
            trajectories_getter=lambda: self._trajectories,
            batch_size=self.config.batch_size,
            num_passes=self.config.n_epochs,
            generator=generator,
            # TODO: add support for n_workers and memory options in config if needed
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

    def on_train_epoch_start(self):
        # Start epoch timer
        train_collector = self.get_rollout_collector("train")
        train_metrics = train_collector.get_metrics()
        self.timings.start("on_train_epoch_start", values=train_metrics)

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

        # Calculate batch losses
        result = self.losses_for_batch(batch, batch_idx)
        
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
        val_collector = self.get_rollout_collector("val")
        val_metrics = val_collector.get_metrics()
        self.timings.start("on_validation_epoch_start", values=val_metrics)

    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Decide if we record a video this eval epoch
        record_video = (
            self.current_epoch == 0
            or self.config.eval_recording_freq_epochs is not None and (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        )

        # Run evaluation with optional recording
        val_env = self.get_env("val")
        video_path = self.run.video_path_for_epoch(self.current_epoch)
        with val_env.recorder(video_path, record_video=record_video):
            # Evaluate using the validation rollout collector to avoid redundant helpers
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
        assert self.run is None, "learn() should only be called once at the start of training"

        from utils.logging import stream_output_to_log

        # Initialize run directory management and convenience Run accessor
        # Initialize run directory (creates runs/<id>/, checkpoints/, and @latest-run symlink)
        self.run = Run.create(
            run_id=wandb.run.id,
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

        # Train the agent
        trainer.fit(self)
    
    # TODO: get this out of here
    def _maybe_warn_observation_policy_mismatch(self):
        from utils.config import Config

        # In case the observation space is RGB, warn if MLP policy is used
        train_env = self.get_env("train")
        is_rgb = train_env.is_rgb_env()
        is_mlp = self.config.policy == Config.PolicyType.mlp
        is_cnn = self.config.policy == Config.PolicyType.cnn
        if is_rgb and is_mlp:
            print(
                "Warning: Detected RGB image observations with MLP policy. "
                "For pixel inputs, consider using CNN for better performance."
            )

        # In case the observation space is not RGB, warn if CNN policy is used
        if not is_rgb and is_cnn:
            print(
                "Warning: Detected non-RGB observations with CNN policy. "
                "For non-image inputs, consider using MLP for better performance."
            )

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
        print_logger = MetricsTableLogger(metrics_monitor=self.metrics_monitor)
        return print_logger
        
    def _build_trainer_loggers(self):
        # Initialize list of loggers
        loggers = []

        # Prepare a wandb logger
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
            PrefitPresentationCallback,
            WandbVideoLoggerCallback,
            WarmupEvalCallback,
        )

        # Initialize callbacks list
        callbacks = []

        # Present config and confirm start at fit start
        callbacks.append(PrefitPresentationCallback())

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
        for key, value in vars(self.config).items():
            if not key.endswith("_schedule") or not value: continue # TODO: extract concern to config
            param = key[: -len("_schedule")]
            assert hasattr(self, param), f"Module {self} has no attribute {param}"
            target_value = getattr(self.config, f"{param}_schedule_target_value", 0.0)
            target_progress = getattr(self.config, f"{param}_schedule_target_progress", 1.0)
            callbacks.append(
                HyperparameterSchedulerCallback(
                    schedule=value,
                    parameter=param,
                    target_value=target_value,
                    target_progress=target_progress,
                    set_value_fn={
                        "policy_lr": self._change_optimizers_lr,
                    }.get(param, None),
                )
            )

        # Checkpointing: save best/last models and metrics
        callbacks.append(ModelCheckpointCallback( # TODO: pass run
            run=self.run,
            metric="val/roll/ep_rew/mean",
            mode="max"
        ))

        # Video logger watches a run-specific media directory lazily (do not create it up-front)
        video_dir = self.run.ensure_video_dir()
        callbacks.append(WandbVideoLoggerCallback(
            media_root=video_dir,
            namespace_depth=1,
        ))

        # If defined in config, early stop after reaching a certain number of vectorized steps
        if self.config.max_timesteps: callbacks.append(
            EarlyStoppingCallback("train/cnt/total_vec_steps", self.config.max_timesteps)
        )

        # If defined in config, early stop when mean training reward reaches a threshold
        train_env = self.get_env("train")
        reward_threshold = train_env.get_return_threshold()
        if self.config.early_stop_on_train_threshold: callbacks.append(EarlyStoppingCallback("train/roll/ep_rew/mean", reward_threshold))

        # If defined in config, early stop when mean validation reward reaches a threshold
        val_env = self.get_env("val")
        reward_threshold = val_env.get_return_threshold()
        if self.config.early_stop_on_eval_threshold: callbacks.append(EarlyStoppingCallback("val/roll/ep_rew/mean", reward_threshold))

        # Also print a terminal summary and alerts recap at the end of training
        callbacks.append(ConsoleSummaryCallback())

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

    def _calc_training_progress(self):
        max_timesteps = self.config.max_timesteps
        if max_timesteps is None: return 0.0
        train_collector = self.get_rollout_collector("train")
        # Use vectorized steps to match the configured step key and early stopping
        total_vec_steps = train_collector.total_vec_steps
        training_progress = max(0.0, min(total_vec_steps / max_timesteps, 1.0))
        return training_progress

    def _change_optimizers_lr(self, lr):
        # Keep attribute in sync for logging/inspection
        self.policy_lr = lr
        optimizers = self.optimizers()
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

    def _log_hyperparameters(self):
        metrics = {
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "clip_range": self.clip_range,
            "policy_lr": self.policy_lr,
        }
        prefixed = {f"hp/{k}": v for k, v in metrics.items()}
        self.metrics_recorder.record("train", prefixed)
