import sys
import time
import types

import torch

# Optional dependency: PyTorch Lightning
try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - lightweight test stub when PL isn't installed
    class _LightningModuleStub:
        def __init__(self, *args, **kwargs):
            pass

        # Methods used indirectly by BaseAgent
        def save_hyperparameters(self, *args, **kwargs):
            pass

        def log_dict(self, *args, **kwargs):
            pass

        def manual_backward(self, *args, **kwargs):
            pass

        def optimizers(self):
            return []

        # Properties accessed in a few places
        @property
        def current_epoch(self):
            return 0

        @property
        def trainer(self):
            # Minimal shape to satisfy .should_stop assignment in tests if reached
            return types.SimpleNamespace(should_stop=False)

    class _PLStub:
        LightningModule = _LightningModuleStub

    pl = _PLStub()  # type: ignore

# Optional decorator (lightweight). Fallback to pass-through if missing.
try:
    from utils.decorators import must_implement
except Exception:  # pragma: no cover
    def must_implement(fn):
        return fn

# Safe/lightweight import
from utils.metrics_buffer import MetricsBuffer


# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    """Base class for RL agents orchestrating envs, rollouts, and training.

    Responsibilities:
    - Build training/eval environments and collectors
    - Provide Lightning hooks for train/eval loops with manual optimization
    - Aggregate and flush metrics efficiently via MetricsBuffer
    - Delegate algorithm-specific pieces to subclasses (models, losses)
    """

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters()

        # Store core attributes
        self.config = config

        # We'll handle optimization manually in training_step
        self.automatic_optimization = False

        # Initialize throughput counters
        self.fit_start_time = None
        self.train_epoch_start_time = None
        self.train_epoch_start_timesteps = None
        self.validation_epoch_start_time = None
        self.validation_epoch_start_timesteps = None

        # Create the environments (lazy import of heavy deps)
        from utils.environment import build_env

        # Training env(s)
        self.train_env = build_env(
            config.env_id,
            seed=config.seed,
            n_envs=config.n_envs,
            subproc=config.subproc,
            obs_type=config.obs_type,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            render_mode=None,
            env_kwargs=config.env_kwargs,
        )

        # Evaluation env (vectorized; separate seed)
        # Use the same level of vectorization as training for fair evaluation when desired.
        # Keep subproc=False to enable video recording.
        self.validation_env = build_env(
            config.env_id,
            seed=config.seed + 1000,
            n_envs=config.n_envs,
            subproc=False,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            obs_type=config.obs_type,
            env_kwargs=config.env_kwargs,
            render_mode="rgb_array",
            record_video=True,
            record_video_kwargs={
                "video_length": 100,  # TODO: softcode this
            },
        )

        # Create required models (subclasses must set self.policy_model)
        self.create_models()
        assert self.policy_model is not None, "Policy model must be created in create_models()"

        # Rollout collectors
        from utils.rollouts import RolloutCollector
        self.train_collector = RolloutCollector(
            self.train_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        self.validation_collector = RolloutCollector(
            self.validation_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        # Shared metrics buffer across epochs
        self._metrics_buffer = MetricsBuffer()

    @must_implement
    def create_models(self):
        pass

    @must_implement
    def losses_for_batch(self, batch, batch_idx):
        pass

    def rollout_collector_hyperparams(self):
        return {
            **self.config.rollout_collector_hyperparams(),
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
        }

    def on_fit_start(self):
        self.fit_start_time = time.time_ns()

    def train_dataloader(self):
        assert self.current_epoch == 0, "train_dataloader should only be called once at the start of training"

        # Collect the first rollout
        self._trajectories = self.train_collector.collect()

        # Use a shared generator for reproducible shuffles across the app
        from utils.random_utils import get_global_torch_generator
        generator = get_global_torch_generator(getattr(self.config, "seed", None))

        # Build efficient index-collate dataloader backed by MultiPassRandomSampler
        # Use a getter to ensure fresh trajectories are used by the collate function
        from utils.dataloaders import build_index_collate_loader_from_collector

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
        # Mark epoch start for instant FPS calculation
        self.train_epoch_start_time = time.time_ns()
        start_metrics = self.train_collector.get_metrics()
        self.train_epoch_start_timesteps = start_metrics["total_timesteps"]

        # Collect fresh trajectories at the start of each training epoch
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

    # TODO: aggregate logging
    def on_train_epoch_end(self):
        rollout_metrics = self.train_collector.get_metrics()
        rollout_metrics.pop("action_distribution", None)

        # TODO: distinguish between instant and start
        # Calculate how many simulation timesteps are being
        # processed per second (includes model training)
        total_timesteps = rollout_metrics["total_timesteps"]
        time_elapsed = max((time.time_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        fps = total_timesteps / time_elapsed
        # Per-epoch instant FPS (since on_train_epoch_start)
        epoch_time_elapsed = max((time.time_ns() - self.train_epoch_start_time) / 1e9, sys.float_info.epsilon)
        epoch_timesteps_elapsed = max(0, total_timesteps - int(self.train_epoch_start_timesteps))
        fps_instant = epoch_timesteps_elapsed / epoch_time_elapsed

        # Include recomputed learning rate and clip range in epoch metrics
        self.log_metrics(
            {
                **rollout_metrics,
                "time_elapsed": time_elapsed,
                "epoch": self.current_epoch,  # TODO: is this the same value as in epoch_start?
                "fps": fps,
                "fps_instant": fps_instant,
            },
            prefix="train",
        )

        self._flush_metrics()

        self._update_schedules()

        # In case we have reached the maximum number of training timesteps then stop training
        if self.config.n_timesteps is not None and total_timesteps >= self.config.n_timesteps:
            print(
                f"Stopping training at epoch {self.current_epoch} with {total_timesteps} timesteps >= limit {self.config.n_timesteps}"
            )
            self.trainer.should_stop = True

    def val_dataloader(self):
        # Validation dataloader is a dummy; actual evaluation uses env rollouts.
        from utils.dataloaders import build_dummy_loader

        return build_dummy_loader()

    def on_validation_epoch_start(self):
        assert (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.config.eval_freq_epochs == 0
        ), (
            f"Validation epoch {self.current_epoch} is not a multiple of eval_freq_epochs {self.config.eval_freq_epochs}"
        )
        self.validation_epoch_start_time = time.time_ns()
        # We'll compute eval FPS based on per-call totals from evaluate_policy
        self.validation_epoch_start_timesteps = 0

    # TODO: check how sb3 does eval_async
    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # New evaluation using vectorized env policy runner
        from utils.evaluation import evaluate_policy

        # Decide if we record a video this eval epoch
        record_video = self.current_epoch == 0 or (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0

        # Use run-specific video directory if available
        video_path = self.run_manager.get_video_dir() / "eval" / "episodes" / f"rollout_epoch_{self.current_epoch}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path = str(video_path)

        # Run evaluation with optional recording
        with self.validation_env.recorder(video_path, record_video=record_video):
            eval_metrics = evaluate_policy(
                self.validation_env,
                self.policy_model,
                n_episodes=int(self.config.eval_episodes),
                deterministic=self.config.eval_deterministic,
            )

        # Frequency check is already enforced in on_validation_epoch_start

        # Calculate FPS
        time_elapsed = max((time.time_ns() - self.validation_epoch_start_time) / 1e9, sys.float_info.epsilon)
        total_timesteps = int(eval_metrics.get("total_timesteps", 0))
        timesteps_elapsed = total_timesteps - self.validation_epoch_start_timesteps
        epoch_fps = int(timesteps_elapsed / time_elapsed)

        # Log metrics
        self.log_metrics({"epoch": int(self.current_epoch), "epoch_fps": epoch_fps, **eval_metrics}, prefix="eval")

        self._flush_metrics()

    def on_validation_epoch_end(self):
        # Validation epoch end is called after all validation steps are done
        # (nothing to do because we already did everything in the validation step)
        pass

    def on_fit_end(self):
        # Log training completion time
        time_elapsed = max((time.time_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")

    def train(self):
        from utils.logging import capture_all_output

        # Ask for confirmation before any heavy setup (keep prior prints grouped)
        if not self._confirm_proceed():
            print("Training aborted by user before start.")
            return

        print("Starting training...")

        # Create wandb logger and run manager
        wandb_logger = self._create_wandb_logger()

        # Setup run directory management
        run_dir, run_logs_dir = self._init_run_manager(wandb_logger)

        print(f"Run directory: {run_dir}")
        print(f"Run ID: {self.run_manager.run_id}")
        print(f"Logs will be saved to: {run_logs_dir}")

        # Set up comprehensive logging using run-specific logs directory
        with capture_all_output(config=self.config, log_dir=run_logs_dir):
            # Save configuration to run directory
            config_path = self.run_manager.save_config(self.config)
            print(f"Configuration saved to: {config_path}")

            # Define step-based metrics to ensure proper ordering
            self._define_wandb_metrics(wandb_logger)

            # Build callbacks and trainer
            callbacks = self._build_callbacks()
            validation_controls = self._get_validation_controls()
            trainer = self._build_trainer(wandb_logger, callbacks, validation_controls)

            trainer.fit(self)

    # -------------------------
    # Helper methods for train()
    # -------------------------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("/", "-").replace("\\", "-")

    def _create_wandb_logger(self):
        from dataclasses import asdict

        from pytorch_lightning.loggers import WandbLogger

        project_name = self.config.project_id if self.config.project_id else self._sanitize_name(self.config.env_id)
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = WandbLogger(project=project_name, name=experiment_name, log_model=True, config=asdict(self.config))
        return wandb_logger

    def _init_run_manager(self, wandb_logger):
        from utils.run_manager import RunManager

        self.run_manager = RunManager()
        run_dir = self.run_manager.setup_run_directory(wandb_logger.experiment)
        run_logs_dir = str(self.run_manager.get_logs_dir())
        return run_dir, run_logs_dir

    def _define_wandb_metrics(self, wandb_logger):
        if wandb_logger.experiment:
            wandb_logger.experiment.define_metric("train/*", step_metric="train/total_timesteps")
            wandb_logger.experiment.define_metric("eval/*", step_metric="train/total_timesteps")

    def _build_callbacks(self):
        # Lazy imports to avoid heavy deps at module import time
        from callbacks import (
            HyperparameterScheduler,
            ModelCheckpointCallback,
            PrintMetricsCallback,
            VideoLoggerCallback,
        )

        # Create video logging callback using run-specific directory
        video_logger_cb = VideoLoggerCallback(
            media_root=str(self.run_manager.get_video_dir()),  # Use run-specific video directory
            namespace_depth=1,
        )

        # Create checkpoint callback using run-specific directory (skip for qlearning)
        checkpoint_cb = None
        if self.config.algo_id != "qlearning":
            checkpoint_cb = ModelCheckpointCallback(
                checkpoint_dir=str(self.run_manager.get_checkpoint_dir()),
                monitor="eval/ep_rew_mean",
                mode="max",
                save_last=True,
                save_threshold_reached=True,
                resume=getattr(self.config, "resume", False),
            )

        # Create algorithm-specific metric rules from metrics config
        from utils.metrics import (
            get_algorithm_metric_rules,
            get_metric_delta_rules,
            get_metric_precision_dict,
        )

        metric_precision = get_metric_precision_dict()
        metric_delta_rules = get_metric_delta_rules()
        algo_metric_rules = get_algorithm_metric_rules(self.config.algo_id)

        # Print metrics callback
        printer_cb = PrintMetricsCallback(
            every_n_steps=200,
            every_n_epochs=10,
            digits=4,
            metric_precision=metric_precision,
            metric_delta_rules=metric_delta_rules,
            algorithm_metric_rules=algo_metric_rules,
        )

        # Hyperparameter scheduler callback
        hyperparam_cb = HyperparameterScheduler(
            control_dir=str(self.run_manager.run_dir / "hyperparam_control"),
            check_interval=2.0,
            enable_lr_scheduling=False,
            enable_manual_control=True,
            verbose=True,
        )

        callbacks = [x for x in [printer_cb, video_logger_cb, checkpoint_cb, hyperparam_cb] if x is not None]
        return callbacks

    def _get_validation_controls(self):
        eval_freq = getattr(self.config, "eval_freq_epochs", None)
        return self._compute_validation_controls(eval_freq)

    @staticmethod
    def _compute_validation_controls(eval_freq_epochs):
        """Pure helper: map eval frequency to PL validation controls.

        Args:
            eval_freq_epochs: int | None
        Returns:
            dict with keys 'limit_val_batches' and 'check_val_every_n_epoch'
        """
        limit_val_batches = 0 if eval_freq_epochs is None else 1.0
        check_val_every_n_epoch = eval_freq_epochs if eval_freq_epochs is not None else 1
        return {
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": check_val_every_n_epoch,
        }

    def _build_trainer(self, wandb_logger, callbacks, validation_controls):
        # Backward-compat shim; delegate to factory. Kept to avoid breaking imports/tests.
        from utils.trainer_factory import build_trainer

        return build_trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            validation_controls=validation_controls,
            max_epochs=self.config.max_epochs,
            accelerator=getattr(self.config, "accelerator", "cpu"),
            devices=getattr(self.config, "devices", None),
        )

    def _backpropagate_and_step(self, losses):
        optimizers = self.optimizers()
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for idx, optimizer in enumerate(optimizers):
            loss = losses[idx]
            optimizer.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm)
            optimizer.step()

    def _get_training_progress(self):
        # TODO: use get_metrics, but make it fast
        total_steps = self.train_collector.total_steps
        total = float(self.config.n_timesteps or 1.0)
        progress = min(max(total_steps / total, 0.0), 1.0)
        return progress

    def _update_schedules(self):
        self._update_schedules__learning_rate()

    # TODO: generalize scheduling support
    def _update_schedules__learning_rate(self):
        if self.config.learning_rate_schedule != "linear":
            return
        progress = self._get_training_progress()
        new_learning_rate = max(self.config.learning_rate * (1.0 - progress), 0.0)
        self._change_optimizers_learning_rate(new_learning_rate)

        # TODO: this should not be logged here
        self.log_metrics({"learning_rate": new_learning_rate}, prefix="train")

    def _change_optimizers_learning_rate(self, learning_rate):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                pg["lr"] = learning_rate

    def log_metrics(self, metrics, *, prefix=None):
        """
        Lightning logger caused significant performance drops, as much as 2x slower train/fps.
        Using custom metric collection / flushing logic to avoid this issue.
        """
        self._metrics_buffer.log(metrics, prefix=prefix)

    def _flush_metrics(self):
        # Flush via buffer abstraction
        self._metrics_buffer.flush_to(self.log_dict)

    # -------------------------
    # Pre-training summary & confirmation helpers
    # -------------------------

    def _print_pretraining_summary(self):
        # Keep output concise and avoid duplication: config and model details were already printed.
        print("\nReview the configuration and model above.")

    def _confirm_proceed(self) -> bool:
        """Ask user to confirm proceeding. Default to Yes on empty input or non-interactive sessions."""
        prompt = "Proceed with training? [Y/n]: "
        try:
            # If not a TTY (e.g., running in CI), default-accept
            import sys
            if not sys.stdin or not sys.stdin.isatty():
                print(f"{prompt}Y (auto)")
                return True
        except Exception:
            # On any introspection failure, default-accept
            print(f"{prompt}Y (auto)")
            return True

        try:
            resp = input(prompt).strip().lower()
        except EOFError:
            # Default to yes if input cannot be read
            print("Y")
            return True
        if resp == "" or resp.startswith("y"):
            return True
        if resp.startswith("n"):
            return False
        # Unrecognized input: default to Yes
        return True
