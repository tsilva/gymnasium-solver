import pytorch_lightning as pl

from torch.nn.utils import clip_grad_norm_

from utils.io import write_json
from utils.timings_tracker import TimingsTracker
from utils.metrics_recorder import MetricsRecorder
from utils.decorators import must_implement
from utils.reports import print_terminal_ascii_summary
from utils.formatting import sanitize_name

CHECKPOINT_PATH = "checkpoints/"

class BaseAgent(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        # TODO: what is this for?
        self.save_hyperparameters()

        # Store experiment configuration
        self.config = config

        # We'll handle optimization manually in training_step
        # (this is needed for multi-model training, eg: policy + value models without shared backbone)
        self.automatic_optimization = False

        # Metrics recorder aggregates per-epoch metrics (train/eval) and maintains
        # a step-aware numeric history for terminal summaries.
        self.metrics = MetricsRecorder(step_key="train/total_timesteps")

        # Initialize timing tracker for training 
        # loop performance measurements
        self.timings = TimingsTracker()

        # TODO: take another look at RunManager vs Run concerns
        self.run_manager = None

        # Create the training environment
        from utils.environment import build_env_from_config
        self.train_env = build_env_from_config(config, seed=config.seed)

        # Create models now that the environment is available. Subclasses use
        # env shapes to build policy/value networks. Must be called before
        # collectors which require self.policy_model.
        self.build_models()

        # Create the rollout collector for the training environment
        from utils.rollouts import RolloutCollector
        self.train_collector = RolloutCollector(
            self.train_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        # Create validation environment and collector
        self.validation_env = build_env_from_config(
            config,
            seed=config.seed + 1000,
            subproc=False,
            render_mode="rgb_array",
            record_video=True,
            record_video_kwargs={
                "video_length": 100,
                "record_env_idx": 0,
            },
        )
        self.validation_collector = RolloutCollector(
            self.validation_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        # Create test environment and collector
        self.test_env = build_env_from_config(
            config,
            seed=config.seed + 2000,
            subproc=False,
            render_mode="rgb_array",
            record_video=True,
            record_video_kwargs={
                "video_length": None,  # full video
                "record_env_idx": 0,
            },
        )
        self.test_collector = RolloutCollector(
            self.test_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        from utils.metrics_triggers import MetricsTriggers
        self.metrics_triggers = MetricsTriggers()


    @must_implement
    def build_models(self):
        # Subclasses must implement this to create their own models (eg: policy, value)
        pass

    @must_implement
    def losses_for_batch(self, batch, batch_idx):
        # Subclasses must implement this to compute the losses for each training steps' batch
        pass

    def rollout_collector_hyperparams(self):
        return {
            **self.config.rollout_collector_hyperparams(),
            "gamma": self.config.gamma, # TODO: shouldn't this be in the config?
            "gae_lambda": self.config.gae_lambda,
            "returns_type": self.config.returns_type,
            "normalize_returns": self.config.normalize_returns == "rollout",
            "advantages_type": self.config.advantages_type,
            "normalize_advantages": self.config.normalize_advantages == "rollout",
        }

    def on_fit_start(self):
        # Start the timing tracker for the entire training run
        self.timings.restart("on_fit_start", steps=0) # TODO: allow tracking arbitrary associated values

        self.metrics_triggers.register_trigger("train/approx_kl", self._check_trigger_1)

    def _check_trigger_1(self):
        history = self.metrics.history()
        # TODO: do code here

    def train_dataloader(self):
        # Some lightweight Trainer stubs used in tests don't manage current_epoch on the module.
        # Guard the assertion to avoid AttributeError while still catching repeated calls.
        assert self.current_epoch == 0, "train_dataloader should only be called once at the start of training"

        # Collect the first rollout
        self._trajectories = self.train_collector.collect()

        # Build efficient index-collate dataloader backed by 
        # MultiPassRandomSampler (allows showing same data N times 
        # per epoch without suffering lightning's epoch turnover costs)
        from utils.dataloaders import build_index_collate_loader_from_collector
        from utils.random import get_global_torch_generator
        generator = get_global_torch_generator(self.config.seed)
        return build_index_collate_loader_from_collector(
            collector=self.train_collector,
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
        total_timesteps = self.train_collector.get_metrics()["total_timesteps"]
        self.timings.restart("on_train_epoch_start", steps=total_timesteps)

        # Log hyperparameters that are tunable in real-time
        self._log_hyperparameters()

        # Collect fresh trajectories at the start of each training epoch
        # Avoid double-collect on the first epoch: train_dataloader() already
        # collected an initial rollout to bootstrap the dataloader. From epoch 1
        # onward, collect once per epoch to ensure constant timestep growth.
        if int(self.current_epoch) > 0:
            self._trajectories = self.train_collector.collect()

    def training_step(self, batch, batch_idx):
        # Calculate batch losses
        losses = self.losses_for_batch(batch, batch_idx)

        # Backpropagate losses and update model
        # parameters according to computed gradients
        self._backpropagate_and_step(losses)

        # We purposely return None here to avoid
        # triggering Lightning's default optimization logic
        # which would interfere with our manual optimization process
        # (we may need to train multiple models with different optimizers))
        return None

    def on_train_epoch_end(self):
        # Update schedules
        self._update_schedules()

        # Check if any registered metric alerts have triggered
        alerts = self.metrics_triggers.check_triggers()
        if alerts:
            print(f"Alerts triggered: {alerts}")

    def val_dataloader(self):
        # TODO: should I just do rollouts here?
        from utils.dataloaders import build_dummy_loader
        return build_dummy_loader()

    def on_validation_epoch_start(self):
        # TODO: why is this being called during warmup epochs?
        # Skip validation entirely during warmup epochs to avoid evaluation overhead
        if not self.should_run_validation_epoch():
            return

        self.timings.restart("on_validation_epoch_start", steps=0)

    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # If eval shouldn't be run this epoch, skip the step (eg: warmup epochs)
        if not self.should_run_validation_epoch(): # TODO: can this be done in the trainer itself?
            return None

        # Decide if we record a video this eval epoch
        record_video = (
            self.current_epoch == 0
            or self.config.eval_recording_freq_epochs is not None and (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        )

        # Run evaluation with optional recording
        checkpoint_dir = self.run_manager.ensure_path(CHECKPOINT_PATH)
        video_path = str(checkpoint_dir / f"epoch={self.current_epoch:02d}.mp4")
        with self.validation_env.recorder(video_path, record_video=record_video):
            # Evaluate using the validation rollout collector to avoid redundant helpers
            val_metrics = self.validation_collector.evaluate_episodes(
                n_episodes=self.config.eval_episodes,
                deterministic=self.config.eval_deterministic,
            )
        
        # Log eval metrics
        total_timesteps = int(val_metrics.get("total_timesteps", 0))
        epoch_fps = self.timings.fps_since("on_validation_epoch_start", steps_now=total_timesteps)
        self.metrics.record("val", {
            **val_metrics,
            "epoch": int(self.current_epoch),
            "epoch_fps": epoch_fps,
        })

    def on_validation_epoch_end(self):
        pass

    def on_fit_end(self):
        # Log training completion time
        time_elapsed = self.timings.seconds_since("on_fit_start")
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes). Reason: {self._early_stop_reason}")

        # TODO: encapsulate in callback
        print_terminal_ascii_summary(self.metrics.history())

        # Record final evaluation video and save associated metrics JSON next to it
        checkpoint_dir = self.run_manager.ensure_path(CHECKPOINT_PATH)
        video_path = checkpoint_dir / "final.mp4"
        with self.test_env.recorder(str(video_path), record_video=True):
            final_metrics = self.test_collector.evaluate_episodes(
                n_episodes=1,
                deterministic=self.config.eval_deterministic,
            )
            json_path = video_path.with_suffix(".json")
            write_json(json_path, final_metrics)
    
    def learn(self):
        assert self.run_manager is None, "learn() should only be called once at the start of training"

        from utils.logging import stream_output_to_log
        from utils.run_manager import RunManager

        wandb_run = self._ensure_wandb_run()
        self.run_manager = RunManager(wandb_run.id)
        
        # Save configuration to run directory
        config_path = self.run_manager.ensure_path("config.json")
        self.config.save_to_json(config_path)
        
        # Set up comprehensive logging using run-specific logs directory
        log_path = self.run_manager.ensure_path("run.log")
        with stream_output_to_log(log_path): self._learn()

    def _learn(self):
        # Prompt user to start training, return if user declines
        if not self._prompt_user_start_training(): return

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
    
    def _prompt_user_start_training(self):
        from utils.user import prompt_confirm
        from utils.logging import display_config_summary

        # TODO: clean this up
        # Collect model details for summary
        try:
            first_param = next(self.policy_model.parameters())
            device_type = first_param.device.type
        except StopIteration:
            device_type = "unknown"

        # Prefer readable enum values where applicable
        def _enum_val(x):
            try:
                return x.value  # Enum
            except Exception:
                return x

        num_params_total = int(sum(p.numel() for p in self.policy_model.parameters()))
        num_params_trainable = int(sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad))

        display_config_summary({
            "Run Details": {
                "Run directory": self.run_manager.get_run_dir(),
                "Run ID": self.run_manager.get_run_id(),
            },
            "Environment Details": {
                "Environment ID": self.train_env.get_id(),
                "Observation type": self.train_env.get_obs_type(),
                "Observation space": self.train_env.observation_space,
                "Action space": self.train_env.action_space,
                "Reward threshold": self.train_env.get_reward_threshold(),
                "Time limit": self.train_env.get_time_limit(),
            },
            "Model Details": {
                "Algorithm": getattr(self.config, "algo_id", None),
                "Policy type": _enum_val(getattr(self.config, "policy", None)),
                "Policy class": type(self.policy_model).__name__,
                "Hidden dims": getattr(self.config, "hidden_dims", None),
                "Activation": getattr(self.config, "activation", None),
                "Device": device_type,
                "Parameters (total)": num_params_total,
                "Parameters (trainable)": num_params_trainable,
            }
        })

        # Ask for confirmation before any heavy setup (keep prior prints grouped)
        # Before prompting, suggest better defaults if we detect mismatches
        self._maybe_warn_observation_policy_mismatch()

        # Prompt if user wants to start training
        start_training = prompt_confirm("Start training?", default=True, quiet=self.config.quiet)
        return start_training

    def _maybe_warn_observation_policy_mismatch(self):
        from utils.config import Config

        # In case the observation space is RGB, warn if MLP policy is used
        is_rgb = self.train_env.is_rgb_env()
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
    
    def should_run_validation_epoch(self) -> bool:
        return self._should_run_eval(self.current_epoch)

    def _should_run_eval(self, epoch_idx: int) -> bool:
        # If freq is None, never evaluate
        freq = self.config.eval_freq_epochs
        if freq is None: return False

        # If warmup is active, skip all epochs <= warmup
        E = epoch_idx + 1
        warmup = self.config.eval_warmup_epochs
        if warmup is not None and E <= warmup: return False

        # Otherwise, evaluate on the cadence grid
        return (E % int(freq)) == 0

    # -------------------------
    # Pre-prompt guidance helpers
    # -------------------------
    
    def _build_trainer_loggers__wandb(self):
        import wandb
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        # Create the wandb logger, attach to the existing run if present
        project_name = self.config.project_id if self.config.project_id else BaseAgent._sanitize_name(self.config.env_id)
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=True,
            config=asdict(self.config),
        ) if wandb.run is None else WandbLogger(log_model=True)

        # Define the common step metric
        wandb_run = wandb_logger.experiment
        wandb_run.define_metric("*", step_metric="train/total_timesteps")

        return wandb_logger
    
    def _build_trainer_loggers__csv(self):
        from loggers.csv_lightning_logger import CsvLightningLogger
        csv_path = self.run_manager.ensure_path("metrics.csv")
        csv_logger = CsvLightningLogger(csv_path=str(csv_path))
        return csv_logger
    
    def _build_trainer_loggers__print(self): # Prepare a terminal print logger that formats metrics from the unified logging stream
        from loggers.print_metrics_logger import PrintMetricsLogger
        # Rely on PrintMetricsLogger defaults sourced from utils.metrics_config
        print_logger = PrintMetricsLogger()
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
            DispatchMetricsCallback,
            ModelCheckpointCallback,
            VideoLoggerCallback,
            EndOfTrainingReportCallback,
            EarlyStoppingCallback,
        )

        # Initialize callbacks list
        callbacks = []

        # Metrics dispatcher: aggregates epoch metrics and logs to Lightning
        callbacks.append(DispatchMetricsCallback())

        # Checkpointing: save best/last models and metrics
        checkpoint_dir = self.run_manager.ensure_path(CHECKPOINT_PATH)
        callbacks.append(ModelCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            metric="val/ep_rew_mean",
            mode="max"
        ))

        # Video logger watches a run-specific media directory lazily (do not create it up-front)
        video_dir = self.run_manager.get_run_dir() / "videos"
        callbacks.append(VideoLoggerCallback(
            media_root=str(video_dir),
            namespace_depth=1,
        ))

        # If defined in config, early stop after reaching a certain number of timesteps
        if self.config.max_timesteps: callbacks.append(
            EarlyStoppingCallback("train/total_timesteps", self.config.max_timesteps)
        )

        # If defined in config, early stop when mean training reward reaches a threshold
        reward_threshold = self.train_env.get_reward_threshold()
        if self.config.early_stop_on_train_threshold: callbacks.append(
            EarlyStoppingCallback("train/ep_rew_mean", reward_threshold)
        )

        # If defined in config, early stop when mean validation reward reaches a threshold
        reward_threshold = self.validation_env.get_reward_threshold()
        if self.config.early_stop_on_eval_threshold: callbacks.append(
            EarlyStoppingCallback("val/ep_rew_mean", reward_threshold)
        )

        # When training ends, write a report describing on the training went
        callbacks.append(EndOfTrainingReportCallback(filename="report.md"))

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
            self.metrics.record("train", metrics)

            # In case a maximum gradient norm is set, 
            # clips gradients so that norm isn't exceeded
            if self.config.max_grad_norm is not None: clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

            # Perform an optimization step 
            # using the computed gradients
            optimizer.step()

    def _calc_training_progress(self):
        max_timesteps = self.config.max_timesteps
        if max_timesteps is None: return 0.0
        total_steps = self.train_collector.total_steps
        training_progress = max(0.0, min(total_steps / max_timesteps, 1.0))
        return training_progress

    def _update_schedules(self):
        self._update_schedules__policy_lr()

    # TODO: generalize scheduling support
    def _update_schedules__policy_lr(self):
        if self.config.policy_lr_schedule != "linear": return
        progress = self._calc_training_progress()
        new_policy_lr = max(self.config.policy_lr * (1.0 - progress), 0.0)
        self._change_optimizers_policy_lr(new_policy_lr)
        # TODO: should I do this here or every epoch?
        # Log scheduled LR under train namespace
        self.metrics.record("train", {"policy_lr": new_policy_lr})

    def _change_optimizers_policy_lr(self, policy_lr):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups: pg["lr"] = policy_lr

    # TODO: review this method
    def configure_optimizers(self):
        from utils.optimizer_factory import build_optimizer
        return build_optimizer(
            params=self.policy_model.parameters(),
            optimizer=self.config.optimizer,
            lr=self.config.policy_lr, # TODO: is this taking annealing into account?
        )

    def get_rollout_collector(self, stage: str):
        return {
            "train": self.train_collector,
            "val": self.validation_collector,
            "test": self.test_collector,
        }[stage]

    def _ensure_wandb_run(self):
        import wandb
        if hasattr(wandb, "run") and wandb.run is not None: return wandb.run
        from dataclasses import asdict
        project_name = self.config.project_id if self.config.project_id else sanitize_name(self.config.env_id)
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb.init(project=project_name, name=experiment_name, config=asdict(self.config))
        return wandb.run

    def _log_hyperparameters(self):
        metrics = {
            "ent_coef": self.config.ent_coef,
            "clip_range": self.config.clip_range,
            "policy_lr": self.config.policy_lr,
        }
        prefixed = {f"hp/{k}": v for k, v in metrics.items()}
        self.metrics.record("train", prefixed)
