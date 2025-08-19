import sys
import time
import torch
from pathlib import Path
import pytorch_lightning as pl

from utils.metrics_buffer import MetricsBuffer
from utils.csv_logger import CsvMetricsLogger
from utils.decorators import must_implement

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
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

        # Training env(s)
        # If using pixel observations, Gymnasium's PixelObservationWrapper requires the
        # base env to be created with render_mode='rgb_array'. Detect that case here.
        #_uses_pixel_obs = False
        #try:
        #    _uses_pixel_obs = any(
        #        isinstance(w, dict) and str(w.get("id")) == "PixelObservationWrapper"
        #        for w in (config.env_wrappers or [])
        #    )
        #except Exception:
        #    _uses_pixel_obs = False
        #train_render_mode = "rgb_array" if _uses_pixel_obs else None

        common_env_kwargs = dict(
            seed=config.seed,
            n_envs = config.n_envs,
            subproc=config.subproc,
            obs_type = config.obs_type,
            frame_stack = config.frame_stack,
            norm_obs = config.normalize_obs,
            env_wrappers=config.env_wrappers,
            env_kwargs=config.env_kwargs,
        )

        # Train environment (vectorized; separate seed)
        from utils.environment import build_env
        self.train_env = build_env(
            config.env_id,
            **dict(
                **common_env_kwargs,
                #seed=config.seed + 1000
            )
        )

        # Evaluation env (vectorized; separate seed)
        # Use the same level of vectorization as training for fair evaluation when desired.
        # Keep subproc=False to enable video recording.
        self.validation_env = build_env(
            config.env_id,
            **dict(
                **common_env_kwargs,
                #seed=config.seed + 2000,
                #subproc=False, # TODO: do I need this?
                render_mode="rgb_array",
                record_video=True,
                record_video_kwargs={
                    # Record full episodes without truncation; the recorder will run
                    # until the evaluation block finishes.
                    "video_length": 100,
                    "record_env_idx": 0,  # Record only first env by default
                }
            )
        )

        # The test environment, this will be used for the 
        # final post train evaluation and video recording
        self.test_env = build_env(
            config.env_id,
            **common_env_kwargs,
            #seed=config.seed + 3000,
            #subproc=False, # TODO: do I need this?
            render_mode="rgb_array",
            record_video=True,
            record_video_kwargs={
                # Record full episodes without truncation; the recorder will run
                # until the evaluation block finishes.
                "video_length": None,
                "record_env_idx": 0,  # Record only first env by default
            }
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

        # Build efficient index-collate dataloader backed by 
        # MultiPassRandomSampler (allows showing same data N times 
        # per epoch without suffering lightning's epoch turnover costs)
        from utils.dataloaders import build_index_collate_loader_from_collector
        from utils.random_utils import get_global_torch_generator
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
        rollout_metrics.pop("action_dist", None)

        # Calculate FPS metrics
        total_timesteps = rollout_metrics["total_timesteps"]
        time_elapsed = max((time.perf_counter_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        fps = total_timesteps / time_elapsed
        epoch_time_elapsed = max((time.perf_counter_ns() - self.train_epoch_start_time) / 1e9, sys.float_info.epsilon)
        epoch_timesteps_elapsed = max(0, total_timesteps - int(self.train_epoch_start_timesteps))
        fps_instant = epoch_timesteps_elapsed / epoch_time_elapsed

        # Log metrics to the buffer
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

        # Flush all metrics logged so far
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
                payload = {"train/action_dist": hist}
                if exp is not None and hasattr(exp, "log"):
                    exp.log(payload, step=step_val)
                else:
                    wandb.log(payload, step=step_val)
        except Exception:
            pass

        self._update_schedules()

    # Early stopping moved to a dedicated callback (see callbacks.EarlyStoppingCallback)

    def val_dataloader(self):
        # TODO: should I just do rollouts here?
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
        # If eval shouldn't be run this epoch, skip the step (eg: warmup epochs)
        if not self._should_run_eval(self.current_epoch):
            return None

        # Decide if we record a video this eval epoch
        record_video = (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        )

        # Run evaluation with optional recording
        checkpoint_dir = self.run_manager.get_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(checkpoint_dir / f"epoch={self.current_epoch:02d}.mp4")
        with self.validation_env.recorder(video_path, record_video=record_video):
            from utils.evaluation import evaluate_policy
            eval_metrics = evaluate_policy(
                self.validation_env,
                self.policy_model,
                n_episodes=int(self.config.eval_episodes),
                deterministic=self.config.eval_deterministic,
            )

        # Calculate FPS
        time_elapsed = max((time.perf_counter_ns() - self.validation_epoch_start_time) / 1e9, sys.float_info.epsilon)
        total_timesteps = int(eval_metrics.get("total_timesteps", 0))
        timesteps_elapsed = total_timesteps - self.validation_epoch_start_timesteps
        epoch_fps = int(timesteps_elapsed / time_elapsed)

        # Log metrics
        if not self.config.log_per_env_eval_metrics:
            eval_metrics = {k: v for k, v in eval_metrics.items() if not k.startswith("per_env/")}

        self.log_metrics({
            "epoch": int(self.current_epoch), 
            "epoch_fps": epoch_fps, 
            **eval_metrics
        }, prefix="eval")

        self._flush_metrics()

    def on_validation_epoch_end(self):
        # Validation epoch end is called after all validation steps are done
        # (nothing to do because we already did everything in the validation step)
        pass

    def on_fit_end(self):
        # Log training completion time
        time_elapsed = max((time.perf_counter_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")

        self._csv_logger.close()
        # Print concise ASCII summary of key metrics for quick inspection
        self._print_terminal_ascii_summary()


    def learn(self):
        from utils.logging import capture_all_output

        # Ask for confirmation before any heavy setup (keep prior prints grouped)
        # Before prompting, suggest better defaults if we detect mismatches
        self._maybe_warn_observation_policy_mismatch()

        self._print_env_spec(self.train_env)

        print("Starting training...")

        # Setup run directory management
        wandb_logger = self._create_wandb_logger()
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
    
    def _print_env_spec(self, env):
        # Show environment details for transparency
        print("\n=== Environment Details ===")
        
        # Observation space and action space from vectorized env
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        # Reward range and threshold when available
        reward_range = env.get_reward_range()
        print(f"Reward range: {reward_range}")

        # Reward threshold if defined
        reward_threshold = env.get_reward_threshold()
        print(f"Reward threshold: {reward_threshold}")
        print("=" * 30)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("/", "-").replace("\\", "-")

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
        # If eval_freq_epochs is None, never evaluate
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

    # -------------------------
    # Pre-prompt guidance helpers
    # -------------------------

    def _maybe_warn_observation_policy_mismatch(self):
        from utils.environment import is_rgb_env
        
        # In case the observation space is RGB, warn if MLP policy is used
        is_rgb = is_rgb_env(self.train_env)
        is_mlp = self.config.policy.lower() == "mlp"
        if is_rgb and is_mlp:
            print(
                "Warning: Detected RGB image observations with MLP policy. "
                "For pixel inputs, consider using CNN for better performance."
            )

        # In case the observation space is not RGB, warn if CNN policy is used
        is_cnn = self.config.policy.lower() == "cnn"
        if not is_rgb and is_cnn:
            print(
                "Warning: Detected non-RGB observations with CNN policy. "
                "For non-image inputs, consider using MLP for better performance."
            )
            
    def _create_wandb_logger(self):
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        project_name = self.config.project_id if self.config.project_id else self._sanitize_name(self.config.env_id)
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = WandbLogger(
            project=project_name, 
            name=experiment_name,
            og_model=True, 
            config=asdict(self.config)
        )
        return wandb_logger

    def _init_run_manager(self, wandb_logger):
        from utils.run_manager import RunManager

        self.run_manager = RunManager()
        run_dir = self.run_manager.setup_run_directory(wandb_logger.experiment)
        run_logs_dir = str(self.run_manager.get_logs_dir())

        self._csv_logger = CsvMetricsLogger(Path(run_dir) / "metrics.csv")
            
        return run_dir, run_logs_dir

    # TODO: review this
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
            HyperparamSyncCallback,
            ModelCheckpointCallback,
            PrintMetricsCallback,
            VideoLoggerCallback,
            EndOfTrainingReportCallback,
            EarlyStoppingCallback,
        )

        callbacks = []

        # Video logger writes to run-specific media directory
        video_logger_cb = VideoLoggerCallback(
            media_root=str(self.run_manager.get_video_dir()),
            namespace_depth=1,
        )
        callbacks.append(video_logger_cb)

        # Checkpointing (skip for qlearning which has no torch model)
        checkpoint_cb = ModelCheckpointCallback(
            checkpoint_dir=str(self.run_manager.get_checkpoint_dir()),
            monitor="eval/ep_rew_mean",
            mode="max",
            save_last=True,
            save_threshold_reached=True,
            resume=self.config.resume,
        )
        callbacks.append(checkpoint_cb)

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

        # TODO: add multi-metric support to EarlyStoppingCallback
        # Early stop after reaching a certain number of timesteps
        earlystop_timesteps_cb = EarlyStoppingCallback(
            "train/total_timesteps",
            self.config.n_timesteps,
            mode="max",
            verbose=True,
        ) if self.config.n_timesteps else None
        if earlystop_timesteps_cb: callbacks.append(earlystop_timesteps_cb)

        # Early stop when mean train reward reaches a threshold
        reward_threshold = self.train_env.get_reward_threshold()
        earlystop_train_reward_cb = EarlyStoppingCallback(
            "train/ep_rew_mean",
            reward_threshold,
            mode="max",
            verbose=True,
        ) if self.config.early_stop_on_train_threshold else None
        if earlystop_train_reward_cb: callbacks.append(earlystop_train_reward_cb)

        # Early stop when mean validation reward reaches a threshold
        reward_threshold = self.validation_env.get_reward_threshold()
        earlystop_eval_reward_cb = EarlyStoppingCallback(
            "eval/ep_rew_mean",
            reward_threshold,
            mode="max",
            verbose=True,
        ) if self.config.early_stop_on_eval_threshold else None
        if earlystop_eval_reward_cb: callbacks.append(earlystop_eval_reward_cb)

        # When training ends, write a report describing on the training went
        report_cb = EndOfTrainingReportCallback(
            filename="report.md"
        )
        callbacks.append(report_cb)

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
        optimizers = self.optimizers()
        if not isinstance(losses, (list, tuple)): losses = [losses]
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]

        for idx, optimizer in enumerate(optimizers):
            loss = losses[idx]
            optimizer.zero_grad()
            self.manual_backward(loss) # TODO: this or loss.backward()?
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
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups: pg["lr"] = learning_rate

    def log_metrics(self, metrics, *, prefix=None):
        """
        Lightning logger caused significant performance drops, as much as 2x slower train/fps.
        Using custom metric collection / flushing logic to avoid this issue.
        """
        self._metrics_buffer.log(metrics, prefix=prefix)

        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()} if prefix else dict(metrics)

        # Update last known step from canonical metric if present
        step_val = prefixed.get("train/total_timesteps")
        if isinstance(step_val, (int, float)):
            self._last_step_for_terminal = int(step_val)

        for k, v in prefixed.items():
            if k.endswith("action_dist"): continue
            if not isinstance(v, (int, float)): continue
            history = self._terminal_history.setdefault(k, [])
            step = self._last_step_for_terminal if k != "train/total_timesteps" else int(v)
            history.append((step, float(v)))

    def _flush_metrics(self):
        # Flush via buffer abstraction
        means = self._metrics_buffer.flush_to(self.log_dict)
        self._csv_logger.log_metrics(means)

    def confirm(prompt: str, default: bool = True, quiet: bool = False) -> bool:
        """Prompt user with yes/no. Defaults on empty, non-interactive, EOF, or quiet mode."""
        yn = "Y/n" if default else "y/N"
        full = f"{prompt} [{yn}]: "

        if quiet:
            print(f"{full}{'Y' if default else 'N'} (quiet)")
            return default

        if not (sys.stdin and sys.stdin.isatty()):
            print(f"{full}{'Y' if default else 'N'} (auto)")
            return default

        try: resp = input(full).strip().lower()
        except EOFError: return default

        return resp.startswith("y") if resp else default
    
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