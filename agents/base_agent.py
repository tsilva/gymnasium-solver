import torch
import pytorch_lightning as pl
import json

from utils.timing import TimingTracker
from utils.metrics_buffer import MetricsBuffer
from utils.metrics_history import MetricsHistory
from utils.decorators import must_implement
from utils.reports import print_terminal_ascii_summary
from utils.torch import compute_param_group_grad_norm

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

        # Buffer used to aggregate per-epoch metrics 
        # (eg: calculate to metric means for logging)
        self._epoch_metrics_buffer = MetricsBuffer()

        # Lightweight history to render ASCII summary at training end
        # Encapsulated in MetricsHistory for clarity and reuse
        self._train_metrics_history = MetricsHistory()

        # Initialize timing tracker for training 
        # loop performance measurements
        self._timing_tracker = TimingTracker()

        self.run_manager = None

        # Create the training environment
        common_env_kwargs = dict(
            seed=config.seed,
            n_envs=config.n_envs,
            max_episode_steps=config.max_episode_steps,
            subproc=config.subproc,
            obs_type=config.obs_type,
            frame_stack=config.frame_stack,
            grayscale_obs=config.grayscale_obs,
            resize_obs=config.resize_obs,
            norm_obs=config.normalize_obs,
            env_wrappers=config.env_wrappers,
            env_kwargs=config.env_kwargs,
        )
        from utils.environment import build_env
        self.train_env = build_env(
            config.env_id,
            **{
                **common_env_kwargs, 
                "seed": config.seed
            },
        )

        # Create models now that the environment is available. Subclasses use
        # env shapes to build policy/value networks. Must be called before
        # collectors which require self.policy_model.
        self.create_models()

        # Create the rollout collector for the training environment
        from utils.rollouts import RolloutCollector
        self.train_collector = RolloutCollector(
            self.train_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        # Create validation environment and collector
        self.validation_env = build_env(
            config.env_id,
            **{
                **common_env_kwargs,
                "seed": config.seed + 1000,  # different seed to minimize correlation
                "subproc": False,
                "render_mode": "rgb_array",
                "record_video": True,
                "record_video_kwargs": {
                    "video_length": 100,  # cap video length to avoid bottlenecks
                    "record_env_idx": 0,
                },
            },
        )
        self.validation_collector = RolloutCollector(
            self.validation_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        # Create test environment and collector
        self.test_env = build_env(
            config.env_id,
            **{
                **common_env_kwargs,
                "seed": config.seed + 2000,
                "subproc": False,
                "render_mode": "rgb_array",
                "record_video": True,
                "record_video_kwargs": {
                    "video_length": None,  # full video
                    "record_env_idx": 0,
                },
            },
        )
        self.test_collector = RolloutCollector(
            self.test_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

    @must_implement
    def create_models(self):
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
        self._timing_tracker.restart("on_fit_start", steps=0) # TODO: allow tracking arbitrary associated values

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
        self._timing_tracker.restart("on_train_epoch_start", steps=total_timesteps)

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
        # Log end of epoch metrics
        self._on_train_epoch_end__log_metrics()

        # Clear the metrics buffer (all callbacks will have logged by now)
        self._epoch_metrics_buffer.clear()

        # Update schedules
        self._update_schedules()

    def _on_train_epoch_end__log_metrics(self):
        # Don't log until we have at least one episode completed 
        # (otherwise we won't be able to get reliable metrics)
        rollout_metrics = self.train_collector.get_metrics()
        total_episodes = rollout_metrics.get("total_episodes", 0)
        if total_episodes == 0: return

        # Global & instant FPS from the same tracker
        total_timesteps = int(rollout_metrics["total_timesteps"])
        time_elapsed = self._timing_tracker.seconds_since("on_fit_start")
        fps_total = self._timing_tracker.fps_since("on_fit_start", steps_now=total_timesteps)
        fps_instant = self._timing_tracker.fps_since("on_train_epoch_start", steps_now=total_timesteps)

        # Prepare metrics to log
        _metrics = {
            **{k:v for k, v in rollout_metrics.items() if not k.endswith("_dist")},
            "time_elapsed": time_elapsed,
            "epoch": self.current_epoch,
            "fps": fps_total,
            "fps_instant": fps_instant,
        }

        # Derive ETA (seconds remaining) from FPS and max_timesteps if available
        if fps_total > 0.0 and self.config.max_timesteps is not None:
            _metrics["eta_s"] = float(self.config.max_timesteps / float(fps_total))
    
        # Log metrics to the buffer
        self.log_metrics(_metrics, prefix="train")

    def val_dataloader(self):
        # TODO: should I just do rollouts here?
        from utils.dataloaders import build_dummy_loader
        return build_dummy_loader()

    def on_validation_epoch_start(self):
        # TODO: why is this being called during warmup epochs?
        # Skip validation entirely during warmup epochs to avoid evaluation overhead
        if not self._should_run_eval(self.current_epoch):
            return

        self._timing_tracker.restart("on_validation_epoch_start", steps=0)

    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # If eval shouldn't be run this epoch, skip the step (eg: warmup epochs)
        if not self._should_run_eval(self.current_epoch): # TODO: can this be done in the trainer itself?
            return None

        # Decide if we record a video this eval epoch
        record_video = (
            self.current_epoch == 0
            or self.config.eval_recording_freq_epochs is not None and (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        )

        # Run evaluation with optional recording
        checkpoint_dir = self.run_manager.ensure_path("checkpoints/")
        video_path = str(checkpoint_dir / f"epoch={self.current_epoch:02d}.mp4")
        with self.validation_env.recorder(video_path, record_video=record_video):
            from utils.evaluation import evaluate_policy
            eval_metrics = evaluate_policy(
                self.validation_env,
                self.policy_model,
                n_episodes=int(self.config.eval_episodes),
                deterministic=self.config.eval_deterministic,
            )
        
        # Log eval metrics
        total_timesteps = int(eval_metrics.get("total_timesteps", 0))
        epoch_fps = self._timing_tracker.fps_since("on_validation_epoch_start", steps_now=total_timesteps)
        self.log_metrics({
            **{k: v for k, v in eval_metrics.items() if not k.startswith("per_env/")},
            "epoch": int(self.current_epoch), 
            "epoch_fps": epoch_fps, 
        }, prefix="eval")

    def on_validation_epoch_end(self):
        # Clear the metrics buffer (all callbacks will have logged by now)
        self._epoch_metrics_buffer.clear()

    def on_fit_end(self):
        # Log training completion time
        time_elapsed = self._timing_tracker.seconds_since("on_fit_start")
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")

        # TODO: encapsulate in callback
        print_terminal_ascii_summary(self._train_metrics_history.as_dict())

        # Record final evaluation video and save associated metrics JSON next to it
        checkpoint_dir = self.run_manager.ensure_path("checkpoints/")
        video_path = checkpoint_dir / "final.mp4"
        with self.test_env.recorder(str(video_path), record_video=True):
            from utils.evaluation import evaluate_policy
            final_metrics = evaluate_policy(
                self.test_env,
                self.policy_model,
                n_episodes=1,
                deterministic=self.config.eval_deterministic,
            )
            json_path = video_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        
    def learn(self):
        assert self.run_manager is None, "learn() should only be called once at the start of training"

        from utils.logging import stream_output_to_log
        from utils.run_manager import RunManager
        
        # Initialize W&B logger and create a run
        wandb_logger = self._create_wandb_logger()
        wandb_run = wandb_logger.experiment

        # Initilize run manager (use W&B run ID as run_id)
        self.run_manager = RunManager(wandb_run.id)
        
        # Save configuration to run directory
        # TODO: pass loggable data instead?
        config_path = self.run_manager.ensure_path("config.json")
        self.config.save_to_json(config_path)
        
        # Set up comprehensive logging using run-specific logs directory
        log_path = self.run_manager.ensure_path("run.log")
        with stream_output_to_log(log_path): self._learn(wandb_logger)

    def _learn(self, wandb_logger):
        # Prompt user to start training, return if user declines
        if not self._prompt_user_start_training(): return

        # Build trainer callbacks
        callbacks = self._build_trainer_callbacks()

        # Build the trainer
        from utils.trainer_factory import build_trainer
        trainer = build_trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            eval_freq_epochs=self.config.eval_freq_epochs,
            eval_warmup_epochs=self.config.eval_warmup_epochs or 0,
        )

        # Train the agent
        trainer.fit(self)
    
    def _prompt_user_start_training(self):
        from utils.user import prompt_confirm
        from utils.logging import display_config_summary

        display_config_summary({
            "Run Details": {
                "Run directory": self.run_manager.get_run_dir(),
                "Run ID": self.run_manager.get_run_id(),
            },
            "Environment Details": {
                "Observation type": self.train_env.get_obs_type(),
                "Observation space": self.train_env.observation_space,
                "Action space": self.train_env.action_space,
                "Reward threshold": self.train_env.get_reward_threshold(),
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
    
    def _create_wandb_logger(self):
        import wandb
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        def _sanitize_name(name: str) -> str:
            return name.replace("/", "-").replace("\\", "-")

        # Create the wandb logger, attach to the existing run if present
        project_name = self.config.project_id if self.config.project_id else _sanitize_name(self.config.env_id)
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
    
    def _build_trainer_callbacks(self):
        """Assemble trainer callbacks, with an optional end-of-training report."""
        # Lazy imports to avoid heavy deps at module import time
        from trainer_callbacks import (
            WandbMetricsLoggerCallback,
            CSVMetricsLoggerCallback,
            PrintMetricsCallback,
            HyperparamSyncCallback,
            ModelCheckpointCallback,
            VideoLoggerCallback,
            EndOfTrainingReportCallback,
            EarlyStoppingCallback,
        )

        # Initialize callbacks list
        callbacks = []

        # CSV Metrics Logger (writes metrics.csv under the run directory)
        csv_path = self.run_manager.ensure_path("metrics.csv")
        callbacks.append(CSVMetricsLoggerCallback(csv_path=str(csv_path)))

        # W&B Logger (logs metrics to W&B)
        callbacks.append(WandbMetricsLoggerCallback())

        # Formatting/precision rules for pretty printing
        from utils.metrics import (
            get_algorithm_metric_rules,
            get_metric_delta_rules,
            get_metric_precision_dict,
        )
        metric_precision = get_metric_precision_dict()
        metric_delta_rules = get_metric_delta_rules()
        algo_metric_rules = get_algorithm_metric_rules(self.config.algo_id)

        # Print metrics once per epoch to align deltas with rollout collection
        printer_cb = PrintMetricsCallback(
            every_n_steps=None,
            every_n_epochs=1,
            digits=4, # TODO: should this be configurable?
            metric_precision=metric_precision,
            metric_delta_rules=metric_delta_rules,
            algorithm_metric_rules=algo_metric_rules,
        )
        callbacks.append(printer_cb)

        # Read hyperparamters from config file (eg: user modified during training)
        hyperparam_sync_cb = HyperparamSyncCallback(
            control_dir=None,
            check_interval=2.0,
            enable_lr_scheduling=False,
            enable_manual_control=True,
            verbose=True,
        )
        callbacks.append(hyperparam_sync_cb)

        # Checkpointing (skip for qlearning which has no torch model)
        checkpoint_dir = self.run_manager.ensure_path("checkpoints/")
        checkpoint_cb = ModelCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            monitor="eval/ep_rew_mean",
            mode="max",
            save_last=True, 
            save_threshold_reached=True,
            resume=False
        )
        callbacks.append(checkpoint_cb)

        # Video logger watches a run-specific media directory lazily (do not create it up-front)
        video_dir = self.run_manager.get_run_dir() / "videos"
        video_logger_cb = VideoLoggerCallback(
            media_root=str(video_dir),
            namespace_depth=1,
        )
        callbacks.append(video_logger_cb)

        # TODO: add multi-metric support to EarlyStoppingCallback
        # Early stop after reaching a certain number of timesteps
        earlystop_timesteps_cb = EarlyStoppingCallback(
            "train/total_timesteps",
            self.config.max_timesteps,
            mode="max",
            verbose=True,
        ) if self.config.max_timesteps else None
        if earlystop_timesteps_cb: callbacks.append(earlystop_timesteps_cb)

        # Early stop when mean training reward reaches a threshold
        reward_threshold = self.train_env.get_reward_threshold()
        earlystop_train_reward_cb = (
            EarlyStoppingCallback(
                "train/ep_rew_mean",
                reward_threshold,
                mode="max",
                verbose=False,
            )
            if self.config.early_stop_on_train_threshold else None
        )
        if earlystop_train_reward_cb: callbacks.append(earlystop_train_reward_cb)

        # Early stop when mean validation reward reaches a threshold
        reward_threshold = self.validation_env.get_reward_threshold()
        earlystop_eval_reward_cb = (
            EarlyStoppingCallback(
                "eval/ep_rew_mean",
                reward_threshold,
                mode="max",
                verbose=False,
            )
            if self.config.early_stop_on_eval_threshold else None
        )
        if earlystop_eval_reward_cb: callbacks.append(earlystop_eval_reward_cb)

        # When training ends, write a report describing on the training went
        report_cb = EndOfTrainingReportCallback(
            filename="report.md"
        )
        callbacks.append(report_cb)

        return callbacks

    def _backpropagate_and_step(self, losses):
        optimizers = self.optimizers()
        if not isinstance(losses, (list, tuple)): losses = [losses]
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]

        for idx, optimizer in enumerate(optimizers):
            loss = losses[idx]
            optimizer.zero_grad()
            self.manual_backward(loss) # TODO: this or loss.backward()?
            
            self._log_policy_grad_norms()

            if self.config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
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
        self.log_metrics({"policy_lr": new_policy_lr}, prefix="train")

    def _change_optimizers_policy_lr(self, policy_lr):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups: pg["lr"] = policy_lr

    # TODO: fix this
    def log_metrics(self, metrics, *, prefix=None):
        """
        Lightning logger caused significant performance drops, as much as 2x slower train/fps.
        Using custom metric collection / flushing logic to avoid this issue.
        """
        # Build a single prefixed mapping and feed both sinks to avoid duplication
        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()} if prefix else dict(metrics)

        # Add to epoch aggregation buffer and terminal history
        self._epoch_metrics_buffer.log(prefixed)
        self._train_metrics_history.update(prefixed)

    # -------------------------
    # Small testable helpers
    # -------------------------

    # TODO: review this method
    def configure_optimizers(self):
        from utils.optimizer_factory import build_optimizer
        return build_optimizer(
            params=self.policy_model.parameters(),
            optimizer=self.config.optimizer,
            lr=self.config.policy_lr, # TODO: is this taking annealing into account?
        )


    def _log_policy_grad_norms(self):
        """Log gradient norms for actor head, critic head, and shared trunk.

        Supports actor-critic models that expose `policy_head` and `value_head`.
        The shared trunk is computed as all parameters excluding head parameters
        (e.g., backbone/CNN feature extractor).
        """
        policy_model = self.policy_model

        # Identify heads if present
        policy_head = policy_model.policy_head
        value_head = policy_model.value_head

        head_param_ids = set()
        actor_params = []
        critic_params = []
        if policy_head is not None:
            actor_params = list(policy_head.parameters())
            head_param_ids.update(id(p) for p in actor_params)
        if value_head is not None:
            critic_params = list(value_head.parameters())
            head_param_ids.update(id(p) for p in critic_params)

        # Trunk are all params not in heads
        trunk_params = [p for p in policy_model.parameters() if id(p) not in head_param_ids]

        metrics = {
            "grad_norm/actor_head": compute_param_group_grad_norm(actor_params),
            "grad_norm/critic_head": compute_param_group_grad_norm(critic_params) if value_head is not None else 0.0,
            "grad_norm/trunk": compute_param_group_grad_norm(trunk_params),
        }
        self.log_metrics(metrics, prefix="train")
