import sys
import time
import types

import torch
from pathlib import Path

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
from utils.csv_logger import CsvMetricsLogger


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

        # CSV metrics logger (initialized with run manager)
        self._csv_logger = None

        # Create the environments (lazy import of heavy deps)
        from utils.environment import build_env

        # Training env(s)
        # If using pixel observations, Gymnasium's PixelObservationWrapper requires the
        # base env to be created with render_mode='rgb_array'. Detect that case here.
        _uses_pixel_obs = False
        try:
            _uses_pixel_obs = any(
                isinstance(w, dict) and str(w.get("id")) == "PixelObservationWrapper"
                for w in (config.env_wrappers or [])
            )
        except Exception:
            _uses_pixel_obs = False

        train_render_mode = "rgb_array" if _uses_pixel_obs else None

        self.train_env = build_env(
            config.env_id,
            seed=config.seed,
            n_envs=config.n_envs,
            subproc=config.subproc,
            obs_type=config.obs_type,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            render_mode=train_render_mode,
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
                # Record full episodes without truncation; the recorder will run
                # until the evaluation block finishes.
                "video_length": None,
                "record_env_idx": 0,  # Record only first env by default
            },
        )

        # Create models now that environments are available. Subclasses use
        # env shapes to build policy/value networks. Must be called before
        # collectors which require self.policy_model.
        self.create_models()

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

        # Lightweight history to render ASCII summary at training end
        # Maps metric name -> list[(step, value)]
        self._terminal_history = {}
        self._last_step_for_terminal = 0

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
        # Use a monotonic clock for durations to avoid NTP/system time jumps
        self.fit_start_time = time.perf_counter_ns()

    def train_dataloader(self):
        assert self.current_epoch == 0, "train_dataloader should only be called once at the start of training"

        # Collect the first rollout
        self._trajectories = self.train_collector.collect()

        # Use a shared generator for reproducible shuffles across the app
        from utils.random_utils import get_global_torch_generator
        generator = get_global_torch_generator(self.config.seed)

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
        self.train_epoch_start_time = time.perf_counter_ns()
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

        # Calculate FPS metrics
        total_timesteps = rollout_metrics["total_timesteps"]
        time_elapsed = max((time.perf_counter_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        fps = total_timesteps / time_elapsed
        epoch_time_elapsed = max((time.perf_counter_ns() - self.train_epoch_start_time) / 1e9, sys.float_info.epsilon)
        epoch_timesteps_elapsed = max(0, total_timesteps - int(self.train_epoch_start_timesteps))
        fps_instant = epoch_timesteps_elapsed / epoch_time_elapsed

        # Log numeric metrics via buffer
        self.log_metrics(
            {
                **rollout_metrics,
                "time_elapsed": time_elapsed,
                "epoch": self.current_epoch,
                "fps": fps,
                "fps_instant": fps_instant,
            },
            prefix="train",
        )

        self._flush_metrics()

        # Log action distribution histogram to W&B, aligned to total_timesteps
        try:
            import wandb  # optional dependency at runtime
            counts = self.train_collector.get_action_histogram_counts(reset=True)
            if counts is not None and counts.sum() > 0:
                num_actions = int(len(counts))
                edges = torch.linspace(-0.5, num_actions - 0.5, steps=num_actions + 1).tolist()
                hist = wandb.Histogram(np_histogram=(counts.tolist(), edges))
                step_val = int(total_timesteps)
                exp = getattr(getattr(self, "logger", None), "experiment", None)
                payload = {"train/action_distribution": hist}
                if exp is not None and hasattr(exp, "log"):
                    exp.log(payload, step=step_val)
                else:
                    wandb.log(payload, step=step_val)
        except Exception:
            pass

        self._update_schedules()

        # Stop condition
        if self.config.n_timesteps is not None and total_timesteps >= self.config.n_timesteps:
            print(
                f"Stopping training at epoch {self.current_epoch} with {total_timesteps} timesteps >= limit {self.config.n_timesteps}"
            )
            trainer = getattr(self, "_trainer", None)
            if trainer is not None and hasattr(trainer, "should_stop"):
                trainer.should_stop = True

    def val_dataloader(self):
        # Validation dataloader is a dummy; actual evaluation uses env rollouts.
        from utils.dataloaders import build_dummy_loader

        return build_dummy_loader()

    def on_validation_epoch_start(self):
        # Skip validation entirely during warmup epochs to avoid evaluation overhead
        if not self._should_run_eval(self.current_epoch):
            # Lightning still calls val hooks when limit_val_batches>0; guard our logic here
            return
        self.validation_epoch_start_time = time.perf_counter_ns()
        # We'll compute eval FPS based on per-call totals from evaluate_policy
        self.validation_epoch_start_timesteps = 0

    # TODO: check how sb3 does eval_async
    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Respect warmup scheduling and frequency
        if not self._should_run_eval(self.current_epoch):
            return None
        # New evaluation using vectorized env policy runner
        from utils.evaluation import evaluate_policy

        # Decide if we record a video this eval epoch
        record_video = (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        )

        # Save eval video alongside checkpoints to simplify correlation
        # Use the same epoch-based naming convention as checkpoints
        ckpt_dir = self.run_manager.get_checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Match the zero-padded epoch format used by checkpoint files: epoch=XX.ckpt
        video_path = ckpt_dir / f"epoch={self.current_epoch:02d}.mp4"
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
        time_elapsed = max((time.perf_counter_ns() - self.validation_epoch_start_time) / 1e9, sys.float_info.epsilon)
        total_timesteps = int(eval_metrics.get("total_timesteps", 0))
        timesteps_elapsed = total_timesteps - self.validation_epoch_start_timesteps
        epoch_fps = int(timesteps_elapsed / time_elapsed)

        # Log metrics
        # Optionally suppress verbose per-env eval diagnostics in logs
        if not self.config.log_per_env_eval_metrics:
            eval_metrics = {k: v for k, v in eval_metrics.items() if not k.startswith("per_env/")}

        self.log_metrics({"epoch": int(self.current_epoch), "epoch_fps": epoch_fps, **eval_metrics}, prefix="eval")

        self._flush_metrics()

    # ----- Evaluation scheduling helpers -----
    def _should_run_eval(self, epoch_idx: int) -> bool:
        """Return True if evaluation should run on this epoch index.

        Scheduling uses 1-based epoch numbers for human-friendly config:
        - Let E = epoch_idx + 1
        - If eval_freq_epochs is None: never evaluate
        - If E < eval_warmup_epochs: skip
        - If E == eval_warmup_epochs: run (first eval after warmup)
        - Otherwise: run when (E - eval_warmup_epochs) % eval_freq_epochs == 0
        """
        freq = self.config.eval_freq_epochs
        if freq is None:
            return False
        warmup = int(self.config.eval_warmup_epochs or 0)
        E = int(epoch_idx) + 1
        if warmup <= 0:
            # First epoch (E==1) then multiples of freq
            return E == 1 or (E % int(freq) == 0)
        # With warmup, first eval at E==warmup, then every freq epochs thereafter
        if E < warmup:
            return False
        return ((E - warmup) % int(freq)) == 0

    def on_validation_epoch_end(self):
        # Validation epoch end is called after all validation steps are done
        # (nothing to do because we already did everything in the validation step)
        pass

    def on_fit_end(self):
        # Log training completion time
        time_elapsed = max((time.perf_counter_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")

        # Print concise ASCII summary of key metrics for quick inspection
        try:
            self._print_terminal_ascii_summary()
        except Exception:
            # Never block program end on summary printing
            pass

    # -------------------------
    # Terminal ASCII summary
    # -------------------------
    def _print_terminal_ascii_summary(self, max_metrics: int = 50, width: int = 48, per_metric_cap: int = 2000):
        """Print an ASCII sparkline summary of recorded numeric metrics.

        Args:
            max_metrics: Maximum number of metrics to print to avoid long outputs.
            width: Target width of sparklines.
            per_metric_cap: Safety cap (ignored here but kept for future trimming consistency).
        """
        history = getattr(self, "_terminal_history", None)
        if not history:
            return

        def downsample(seq, target):
            if len(seq) <= target:
                return seq
            # Uniform uniform sampling by index
            step = len(seq) / float(target)
            return [seq[int(i * step)] for i in range(target)]

        def spark(values, w):
            blocks = "▁▂▃▄▅▆▇█"
            if not values:
                return ""
            vmin = min(values)
            vmax = max(values)
            if vmax == vmin:
                return "─" * max(1, min(w, len(values)))
            data = downsample(values, max(1, w))
            out = []
            rng = (vmax - vmin) or 1.0
            for v in data:
                idx = int((v - vmin) / rng * (len(blocks) - 1))
                out.append(blocks[max(0, min(idx, len(blocks) - 1))])
            return "".join(out)

        # Prefer train/* then eval/* then others for readability
        keys = sorted(history.keys(), key=lambda k: (0 if k.startswith("train/") else 1 if k.startswith("eval/") else 2, k))
        shown = 0
        printed_header = False
        for k in keys:
            pts = history.get(k) or []
            if len(pts) < 2:
                continue
            # Collapse duplicate steps (keep last)
            by_step = {}
            for s, v in pts:
                by_step[int(s)] = float(v)
            if not by_step:
                continue
            steps_sorted = sorted(by_step)
            values = [by_step[s] for s in steps_sorted]
            chart = spark(values, width)
            vmin = min(values)
            vmax = max(values)
            vlast = values[-1]
            if not printed_header:
                print("\n=== Metrics Summary (ASCII) ===")
                printed_header = True
            print(f"{k:>26}: {chart}  min={vmin:.4g} max={vmax:.4g} last={vlast:.4g}")
            shown += 1
            if shown >= max_metrics:
                break
        if printed_header:
            if shown == 0:
                print("(no numeric metrics to summarize)")
            print("=" * 30)

    def learn(self):
        from utils.logging import capture_all_output

        # Ask for confirmation before any heavy setup (keep prior prints grouped)
        # Before prompting, suggest better defaults if we detect mismatches
        self._maybe_warn_mlp_on_rgb_obs()
        # Show environment details for transparency
        try:
            print("\n=== Environment Details ===")
            # Observation space and action space from vectorized env
            obs_space = getattr(self.train_env, "observation_space", None)
            act_space = getattr(self.train_env, "action_space", None)
            if obs_space is not None:
                print(f"Observation space: {obs_space}")
            if act_space is not None:
                print(f"Action space: {act_space}")

            # Reward range and threshold when available
            reward_range = None
            if hasattr(self.train_env, "get_reward_range"):
                try:
                    reward_range = self.train_env.get_reward_range()
                except Exception:
                    reward_range = None
            if reward_range is None and hasattr(getattr(self.train_env, "envs", [None])[0], "reward_range"):
                rr = getattr(self.train_env.envs[0], "reward_range", None)
                if isinstance(rr, (tuple, list)) and len(rr) == 2:
                    reward_range = tuple(rr)
            if reward_range is not None:
                print(f"Reward range: {reward_range}")

            reward_threshold = None
            if hasattr(self.train_env, "get_reward_threshold"):
                try:
                    reward_threshold = self.train_env.get_reward_threshold()
                except Exception:
                    reward_threshold = None
            if reward_threshold is not None:
                print(f"Reward threshold: {reward_threshold}")
            print("=" * 30)
        except Exception:
            # Never block training if introspection fails
            pass
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

    # Backward-compatibility: some tests and callers expect a fit() method on
    # the agent (mirroring Lightning's Trainer.fit). Provide a thin alias so
    # agent.fit() behaves the same as learn().
    def fit(self):  # pragma: no cover - covered indirectly by integration test
        return self.learn()

    # -------------------------
    # Helper methods for train()
    # -------------------------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("/", "-").replace("\\", "-")

    # -------------------------
    # Pre-prompt guidance helpers
    # -------------------------

    def _maybe_warn_mlp_on_rgb_obs(self):
        """If obs space appears to be RGB images and policy is MLP, print a warning.

        Intent: gently nudge users to use CnnPolicy when training from pixels.
        Triggered right before the start-training confirmation prompt.
        """
        try:
            policy_type = getattr(self.config, "policy", "MlpPolicy")
            # Normalize common strings
            is_mlp = False
            if isinstance(policy_type, str):
                s = policy_type.lower()
                is_mlp = s in ("mlppolicy", "mlp", "mlp_policy")
            # If policy was provided as a class, heuristically treat non-CNN as MLP
            else:
                is_mlp = True

            if not is_mlp:
                return

            obs_space = getattr(self.train_env, "observation_space", None)
            if obs_space is None:
                return

            # Heuristic: RGB observations are typically uint8 Box with 3 channels stacked
            import numpy as np
            from gymnasium import spaces

            if not isinstance(obs_space, spaces.Box):
                return

            shape = tuple(getattr(obs_space, "shape", ()) or ())
            if len(shape) < 3:
                return

            # VecFrameStack may multiply channels; dtype uint8 is a strong signal for pixels
            is_uint8 = getattr(obs_space, "dtype", None) == np.uint8
            channels_like = shape[-1]
            looks_rgb = is_uint8 and (channels_like == 3 or channels_like % 3 == 0)

            if not looks_rgb:
                return

            print(
                "Warning: Detected RGB image observations with MlpPolicy. "
                "For pixel inputs, consider using CnnPolicy for better performance."
            )
        except Exception:
            # Never block training on advisory messages
            pass

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
        # Initialize high-throughput CSV metrics logger at run root
        try:
            self._csv_logger = CsvMetricsLogger(Path(run_dir) / "metrics.csv")
        except Exception:
            self._csv_logger = None  # Never block training on CSV setup
        return run_dir, run_logs_dir

    def _define_wandb_metrics(self, wandb_logger):
        if wandb_logger.experiment:
            wandb_logger.experiment.define_metric("train/*", step_metric="train/total_timesteps")
            wandb_logger.experiment.define_metric("train/hyperparams/*", step_metric="train/total_timesteps")
            wandb_logger.experiment.define_metric("eval/*", step_metric="train/total_timesteps")
            # Expose metric bounds to the W&B run config for dashboard consumption
            try:
                from utils.metrics import get_metric_bounds
                bounds = get_metric_bounds()
                if isinstance(bounds, dict) and bounds:
                    # Store under a dedicated namespace to avoid clutter
                    wandb_logger.experiment.config.update({"metric_bounds": bounds}, allow_val_change=True)
            except Exception:
                # Never block training on telemetry metadata
                pass

    def _build_callbacks(self):
        """Assemble trainer callbacks, with an optional end-of-training report."""
        # Lazy imports to avoid heavy deps at module import time
        from callbacks import (
            HyperparameterScheduler,
            ModelCheckpointCallback,
            PrintMetricsCallback,
            VideoLoggerCallback,
        )
        # Optional report callback: import may fail in minimal environments
        try:
            from callbacks.end_of_training_report import EndOfTrainingReportCallback  # type: ignore
        except Exception:
            EndOfTrainingReportCallback = None  # type: ignore

        # Video logger writes to run-specific media directory
        video_logger_cb = VideoLoggerCallback(
            media_root=str(self.run_manager.get_video_dir()),
            namespace_depth=1,
        )

        # Checkpointing (skip for qlearning which has no torch model)
        checkpoint_cb = None
        if self.config.algo_id != "qlearning":
            checkpoint_cb = ModelCheckpointCallback(
                checkpoint_dir=str(self.run_manager.get_checkpoint_dir()),
                monitor="eval/ep_rew_mean",
                mode="max",
                save_last=True,
                save_threshold_reached=True,
                resume=self.config.resume,
            )

        # Formatting/precision rules for pretty printing
        from utils.metrics import (
            get_algorithm_metric_rules,
            get_metric_delta_rules,
            get_metric_precision_dict,
        )
        metric_precision = get_metric_precision_dict()
        metric_delta_rules = get_metric_delta_rules()
        algo_metric_rules = get_algorithm_metric_rules(self.config.algo_id)

        printer_cb = PrintMetricsCallback(
            every_n_steps=200,
            every_n_epochs=10,
            digits=4,
            metric_precision=metric_precision,
            metric_delta_rules=metric_delta_rules,
            algorithm_metric_rules=algo_metric_rules,
        )

        hyperparam_cb = HyperparameterScheduler(
            control_dir=None,
            check_interval=2.0,
            enable_lr_scheduling=False,
            enable_manual_control=True,
            verbose=True,
        )

        report_cb = EndOfTrainingReportCallback(filename="report.md") if EndOfTrainingReportCallback else None

        callbacks = [x for x in [printer_cb, video_logger_cb, checkpoint_cb, hyperparam_cb, report_cb] if x is not None]
        return callbacks

    def _get_validation_controls(self):
        # Keep Lightning validation cadence driven by eval_freq_epochs; warmup is enforced in hooks.
        eval_freq = self.config.eval_freq_epochs
        warmup = self.config.eval_warmup_epochs or 0
        # If warmup is active, request validation every epoch and gate in hooks
        eff_freq = 1 if (eval_freq is not None and warmup > 0) else eval_freq
        return self._compute_validation_controls(eff_freq)

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
            accelerator=self.config.accelerator,
            devices=self.config.devices,
        )

    def _backpropagate_and_step(self, losses):
        # Try to get optimizers from Lightning; if not attached to a Trainer
        # (as in our lightweight integration tests), fall back to a cached
        # manual optimizer created via configure_optimizers().
        try:
            optimizers = self.optimizers()
        except Exception:
            optimizers = getattr(self, "_manual_optimizers", None)
            if optimizers is None:
                optimizers = self.configure_optimizers()
                if not isinstance(optimizers, (list, tuple)):
                    optimizers = [optimizers]
                self._manual_optimizers = optimizers

        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        for idx, optimizer in enumerate(optimizers):
            loss = losses[idx]
            optimizer.zero_grad()
            # Use Lightning's manual_backward when available; otherwise, raw autograd
            try:
                self.manual_backward(loss)
            except Exception:
                loss.backward()
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
        # Log scheduled LR under hyperparams namespace
        self.log_metrics({"learning_rate": new_learning_rate}, prefix="train/hyperparams")

    def _change_optimizers_learning_rate(self, learning_rate):
        # Obtain optimizers from Lightning when attached; otherwise, fall back
        # to manual optimizers created via configure_optimizers().
        try:
            optimizers = self.optimizers()
        except Exception:
            optimizers = getattr(self, "_manual_optimizers", None)
            if optimizers is None:
                optimizers = self.configure_optimizers()
                if not isinstance(optimizers, (list, tuple)):
                    optimizers = [optimizers]
                self._manual_optimizers = optimizers
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

        # Capture lightweight numeric history for terminal ASCII summary
        try:
            if prefix:
                prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
            else:
                prefixed = dict(metrics)

            # Update last known step from canonical metric if present
            step_val = prefixed.get("train/total_timesteps")
            if isinstance(step_val, (int, float)):
                self._last_step_for_terminal = int(step_val)

            for k, v in prefixed.items():
                if k.endswith("action_distribution"):
                    continue
                if not isinstance(v, (int, float)):
                    continue
                history = self._terminal_history.setdefault(k, [])
                step = self._last_step_for_terminal if k != "train/total_timesteps" else int(v)
                history.append((step, float(v)))
        except Exception:
            # Never fail due to bookkeeping issues
            pass

    def _flush_metrics(self):
        # Flush via buffer abstraction
        means = self._metrics_buffer.flush_to(self.log_dict)
        # Also enqueue to CSV logger (asynchronous, non-blocking)
        try:
            if self._csv_logger is not None:
                self._csv_logger.log_metrics(means)
        except Exception:
            pass

    # Ensure CSV logger is closed cleanly when training ends
    def on_fit_end(self):
        try:
            if self._csv_logger is not None:
                self._csv_logger.close()
        except Exception:
            pass
        # Call super if available
        try:
            return super().on_fit_end()
        except Exception:
            return None

    # -------------------------
    # Pre-training summary & confirmation helpers
    # -------------------------

    def _print_pretraining_summary(self):
        # Keep output concise and avoid duplication: config and model details were already printed.
        print("\nReview the configuration and model above.")

    def _confirm_proceed(self) -> bool:
        """Ask user to confirm proceeding. Default to Yes on empty input or non-interactive sessions."""
        # Honor quiet mode to avoid any interactive prompts
        try:
            if getattr(getattr(self, "config", object()), "quiet", False):
                print("Proceed with training? [Y/n]: Y (quiet)")
                return True
        except Exception:
            pass
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
