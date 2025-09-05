import torch
import pytorch_lightning as pl
import json

from utils.metrics_buffer import MetricsBuffer
from utils.decorators import must_implement
from utils.reports import print_terminal_ascii_summary

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

        # Shared metrics buffer across epochs
        self._metrics_buffer = MetricsBuffer()

        # Initialize timing tracker for training 
        # loop performance measurements
        from utils.timing import TimingTracker
        self.timing = TimingTracker()

        # Lightweight history to render ASCII summary at training end
        # Maps metric name -> list[(step, value)]
        # TODO: move this inside callback
        self._terminal_history = {}
        self._last_step_for_terminal = 0

        self.run_manager = None

        # Track best episode reward means seen so far for train/eval
        # Populated when logging epoch metrics so they can be surfaced in CSV/console
        self._best_rewards = {"train": None, "eval": None}

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

        # Validation/Test envs
        # Retro (Gym Retro) cannot have multiple emulator instances per process.
        # To respect this, we create evaluation envs lazily (just-in-time) and
        # close them immediately after use during validation/test for Retro.
        try:
            from utils.environment import is_stable_retro_env_id as _is_retro_fn  # type: ignore
        except Exception:
            _is_retro_fn = lambda _eid: False
        _is_retro = False
        try:
            _is_retro = bool(_is_retro_fn(config.env_id))
        except Exception:
            _is_retro = False

        if not _is_retro:
            # Standard path: keep persistent eval/test envs with video recorder wrapper
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
        else:
            # Retro path: do not create envs now; create/close them on demand.
            # Provide a minimal stub for callbacks that query reward thresholds.
            class _EvalEnvStub:
                def get_reward_threshold(self):
                    return None

            self.validation_env = _EvalEnvStub()
            self.validation_collector = None  # created lazily if needed
            self.test_env = _EvalEnvStub()
            self.test_collector = None

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
        self.timing.restart("on_fit_start", steps=0) # TODO: allow tracking arbitrary associated values

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
        self.timing.restart("on_train_epoch_start", steps=total_timesteps)

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

    # TODO: aggregate logging
    def on_train_epoch_end(self):
        # Pull cumulative steps once
        rollout_metrics = self.train_collector.get_metrics()
        rollout_metrics.pop("action_dist", None)

        # Global & instant FPS from the same tracker
        total_timesteps = int(rollout_metrics["total_timesteps"])
        time_elapsed = self.timing.seconds_since("on_fit_start")
        fps_total = self.timing.fps_since("on_fit_start", steps_now=total_timesteps)
        fps_instant = self.timing.fps_since("on_train_epoch_start", steps_now=total_timesteps)

        # If no episodes have completed yet, avoid logging ep_*_mean metrics
        # to external sinks (e.g., W&B/CSV) to prevent an initial zero spike.
        # Keep immediate last stats available for local use/tests.
        try:
            if len(self.train_collector.episode_reward_deque) == 0:
                rollout_metrics.pop("ep_rew_mean", None)
                rollout_metrics.pop("ep_len_mean", None)
        except Exception:
            # Be defensive: if collector lacks the deque, proceed without pruning
            pass

        # Prepare metrics to log
        _metrics = {
            **rollout_metrics,
            "time_elapsed": time_elapsed,
            "epoch": self.current_epoch,
            "fps": fps_total,  # run-wide average FPS
            "fps_instant": fps_instant,
        }

        # Derive ETA (seconds remaining) from FPS and max_timesteps if available
        try:
            max_ts = float(self.config.max_timesteps) if self.config.max_timesteps is not None else None
        except Exception:
            max_ts = None
        if max_ts is not None:
            try:
                remaining = max(0.0, float(max_ts) - float(total_timesteps))
                if float(fps_total) > 0.0:
                    _metrics["eta_s"] = float(remaining / float(fps_total))
            except Exception:
                # Be robust: skip ETA if any cast/division fails
                pass

        # Log metrics to the buffer
        self.log_metrics(_metrics, prefix="train")

        #self._flush_metrics()

        self._update_schedules()

    def val_dataloader(self):
        # TODO: should I just do rollouts here?
        from utils.dataloaders import build_dummy_loader
        return build_dummy_loader()

    def on_validation_epoch_start(self):
        # Skip validation entirely during warmup epochs to avoid evaluation overhead
        if not self._should_run_eval(self.current_epoch):
            return

        self.timing.restart("on_validation_epoch_start", steps=0)

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
            or self.config.eval_recording_freq_epochs is not None and (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        )

        # Run evaluation with optional recording
        checkpoint_dir = self.run_manager.ensure_path("checkpoints/")
        video_path = str(checkpoint_dir / f"epoch={self.current_epoch:02d}.mp4")

        # Detect Retro to use a temporary env
        try:
            from utils.environment import is_stable_retro_env_id as _is_retro_fn  # type: ignore
        except Exception:
            _is_retro_fn = lambda _eid: False
        use_temp_env = False
        try:
            use_temp_env = bool(_is_retro_fn(self.config.env_id))
        except Exception:
            use_temp_env = False

        if use_temp_env:
            # Build a temporary single-env dummy vec with recording enabled, then close
            from utils.environment import build_env
            temp_env = build_env(
                self.config.env_id,
                seed=self.config.seed + 1000,
                n_envs=1,
                subproc=False,
                render_mode="rgb_array",
                record_video=True,
                record_video_kwargs={
                    "video_length": 100,
                    "record_env_idx": 0,
                },
                env_wrappers=self.config.env_wrappers,
                env_kwargs=self.config.env_kwargs,
                obs_type=self.config.obs_type,
                frame_stack=self.config.frame_stack,
                grayscale_obs=self.config.grayscale_obs,
                resize_obs=self.config.resize_obs,
                norm_obs=self.config.normalize_obs,
            )
            try:
                with temp_env.recorder(video_path, record_video=record_video):
                    from utils.evaluation import evaluate_policy
                    eval_metrics = evaluate_policy(
                        temp_env,
                        self.policy_model,
                        n_episodes=int(self.config.eval_episodes),
                        deterministic=self.config.eval_deterministic,
                        max_steps_per_episode=1000,
                    )
            finally:
                try:
                    temp_env.close()
                except Exception:
                    pass
        else:
            with self.validation_env.recorder(video_path, record_video=record_video):
                from utils.evaluation import evaluate_policy
                eval_metrics = evaluate_policy(
                    self.validation_env,
                    self.policy_model,
                    n_episodes=int(self.config.eval_episodes),
                    deterministic=self.config.eval_deterministic,
                )
        
        total_timesteps = int(eval_metrics.get("total_timesteps", 0))
        epoch_fps = self.timing.fps_since("on_validation_epoch_start", steps_now=total_timesteps)

        # Log metrics
        #if not self.config.log_per_env_eval_metrics:
        eval_metrics = {k: v for k, v in eval_metrics.items() if not k.startswith("per_env/")}

        # If evaluation produced zero episodes (edge cases), avoid logging
        # ep_*_mean to external sinks to prevent an initial zero spike.
        try:
            if int(eval_metrics.get("total_episodes", 0)) <= 0:
                eval_metrics.pop("ep_rew_mean", None)
                eval_metrics.pop("ep_len_mean", None)
        except Exception:
            pass

        self.log_metrics({
            "epoch": int(self.current_epoch), 
            "epoch_fps": epoch_fps, 
            **eval_metrics
        }, prefix="eval")

    def on_validation_epoch_end(self):
        # Validation epoch end is called after all validation steps are done
        # (nothing to do because we already did everything in the validation step)
        pass

    def on_fit_end(self):
        # Log training completion time
        time_elapsed = self.timing.seconds_since("on_fit_start")
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")

        # TODO: encapsulate in callback
        history = getattr(self, "_terminal_history", None)
        if history: print_terminal_ascii_summary(history)

        # Record final evaluation video and save associated metrics JSON next to it
        checkpoint_dir = self.run_manager.ensure_path("checkpoints/")
        video_path = checkpoint_dir / "final.mp4"
        final_metrics = None

        # If validation is disabled (eval_freq_epochs is None or <= 0), skip final evaluation/video
        try:
            _freq = getattr(self.config, "eval_freq_epochs", None)
            _eval_disabled = (_freq is None) or (int(_freq) <= 0)
        except Exception:
            _eval_disabled = True

        if not _eval_disabled:
            # Detect Retro and evaluate with a temporary env to avoid multi-instance limit
            try:
                from utils.environment import is_stable_retro_env_id as _is_retro_fn  # type: ignore
            except Exception:
                _is_retro_fn = lambda _eid: False
            use_temp_env = False
            try:
                use_temp_env = bool(_is_retro_fn(self.config.env_id))
            except Exception:
                use_temp_env = False

            if use_temp_env:
                from utils.environment import build_env
                temp_env = build_env(
                    self.config.env_id,
                    seed=self.config.seed + 2000,
                    n_envs=1,
                    subproc=False,
                    render_mode="rgb_array",
                    record_video=True,
                    record_video_kwargs={
                        "video_length": None,
                        "record_env_idx": 0,
                    },
                    env_wrappers=self.config.env_wrappers,
                    env_kwargs=self.config.env_kwargs,
                    obs_type=self.config.obs_type,
                    frame_stack=self.config.frame_stack,
                    grayscale_obs=self.config.grayscale_obs,
                    resize_obs=self.config.resize_obs,
                    norm_obs=self.config.normalize_obs,
                )
                try:
                    with temp_env.recorder(str(video_path), record_video=True):
                        from utils.evaluation import evaluate_policy
                        final_metrics = evaluate_policy(
                            temp_env,
                            self.policy_model,
                            n_episodes=1,
                            deterministic=self.config.eval_deterministic,
                            max_steps_per_episode=2000,
                        )
                finally:
                    try:
                        temp_env.close()
                    except Exception:
                        pass
            else:
                with self.test_env.recorder(str(video_path), record_video=True):
                    from utils.evaluation import evaluate_policy
                    final_metrics = evaluate_policy(
                        self.test_env,
                        self.policy_model,
                        n_episodes=1,
                        deterministic=self.config.eval_deterministic,
                    )
        # Always attempt to write metrics JSON alongside the final video
        try:
            if final_metrics is not None:
                json_path = video_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(final_metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Ensure a final checkpoint exists even if validation was disabled and no
        # eval checkpoints were produced. Also create convenient last/best symlinks
        # for both checkpoint and associated final video/json.
        try:
            from trainer_callbacks.model_checkpoint import ModelCheckpointCallback
            # If there are no .ckpt files in the run's checkpoint directory, save one now
            has_ckpt = any(p.suffix == ".ckpt" for p in checkpoint_dir.glob("*.ckpt"))
            if not has_ckpt:
                # Save a single snapshot checkpoint containing model and optimizer states
                cb = ModelCheckpointCallback(checkpoint_dir=str(checkpoint_dir))
                # Use a stable basename that won't conflict with epoch-based names
                saved_path = checkpoint_dir / "final.ckpt"
                cb._save_checkpoint(  # type: ignore[attr-defined]
                    self,
                    saved_path,
                    is_best=False,
                    is_last=True,
                    is_threshold=False,
                    metrics=final_metrics if isinstance(final_metrics, dict) else None,
                    current_eval_reward=(
                        float(final_metrics.get("ep_rew_mean"))
                        if isinstance(final_metrics, dict) and "ep_rew_mean" in final_metrics
                        else None
                    ),
                    threshold_value=None,
                )
                # Maintain last/best symlinks for convenience
                try:
                    cb._update_symlink(checkpoint_dir / "last.ckpt", saved_path)  # type: ignore[attr-defined]
                except Exception:
                    pass
                # If there was no best.ckpt yet, mirror last->best
                try:
                    best_ckpt = checkpoint_dir / "best.ckpt"
                    if not best_ckpt.exists() and not best_ckpt.is_symlink():
                        cb._update_symlink(best_ckpt, saved_path)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Create video/json best/last symlinks pointing to final artifacts when missing
            try:
                def _link(src_rel: str, dst: str):
                    try:
                        src = checkpoint_dir / src_rel
                        dstp = checkpoint_dir / dst
                        if not dstp.exists() and not dstp.is_symlink() and src.exists():
                            ModelCheckpointCallback._update_symlink(dstp, src)  # type: ignore[attr-defined]
                    except Exception:
                        pass

                _link("final.mp4", "last.mp4")
                _link("final.mp4", "best.mp4")
                _link("final.json", "last.json")
                _link("final.json", "best.json")
            except Exception:
                pass
        except Exception:
            # Never fail training shutdown due to checkpoint finalization issues
            pass
        
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
        config_path = self.run_manager.ensure_path("config.json")
        self.config.save_to_json(config_path)
        
        # Set up comprehensive logging using run-specific logs directory
        log_path = self.run_manager.ensure_path("run.log")
        with stream_output_to_log(log_path): self._learn(wandb_logger)

    def _learn(self, wandb_logger):
        # Prompt user to start training, return if user declines
        if not self._prompt_user_start_training(): return

        # Build callbacks and trainer
        callbacks = self._build_trainer_callbacks()

        # Keep Lightning validation cadence driven by eval_freq_epochs; warmup is enforced in hooks.
        eval_freq = self.config.eval_freq_epochs
        warmup = self.config.eval_warmup_epochs or 0

        # Treat non-positive values as disabled
        try:
            if eval_freq is not None and int(eval_freq) <= 0:
                eval_freq = None
        except Exception:
            eval_freq = None

        # If warmup is active, request validation every epoch and gate in hooks
        eval_freq_epochs = 1 if (eval_freq is not None and warmup > 0) else eval_freq
        limit_val_batches = 0 if eval_freq_epochs is None else 1.0
        check_val_every_n_epoch = eval_freq_epochs if eval_freq_epochs is not None else 1

        trainer = self._build_trainer(
            wandb_logger, 
            callbacks, 
            {
                "limit_val_batches": limit_val_batches,
                "check_val_every_n_epoch": check_val_every_n_epoch,
            }
        )
        trainer.fit(self)
    
    def _prompt_user_start_training(self):
        from utils.user import prompt_confirm

        print("\n=== Run Details ===")
        print(f"Run directory: {self.run_manager.get_run_dir()}")
        print(f"Run ID: {self.run_manager.get_run_id()}")
        print("=" * 30)

        print("\n=== Environment Details ===")
        if hasattr(self.train_env, 'print_spec'):
            self.train_env.print_spec()
        else:
            # Minimal fallback for stub env used in tests
            try:
                print(f"n_envs: {getattr(self.train_env, 'num_envs', '?')}")
                print(f"input_dim: {getattr(self, 'input_dim', '?')}")
                print(f"output_dim: {getattr(self, 'output_dim', '?')}")
            except Exception:
                pass
        print("=" * 30)

        # Also log configuration details for reproducibility
        try:
            import utils.logging as _logging_mod
            _log_fn = getattr(_logging_mod, "log_config_details", None)
            if callable(_log_fn):
                _log_fn(self.config)
        except Exception:
            pass

        # Ask for confirmation before any heavy setup (keep prior prints grouped)
        # Before prompting, suggest better defaults if we detect mismatches
        self._maybe_warn_observation_policy_mismatch()

        # Prompt if user wants to start training
        start_training = prompt_confirm("Start training?", default=True, quiet=self.config.quiet)
        return start_training

    def _maybe_warn_observation_policy_mismatch(self):
        # utils.environment may be monkeypatched in tests without is_rgb_env. Be robust.
        try:
            import utils.environment as _env_mod  # type: ignore
        except Exception:
            return

        is_rgb_env_fn = getattr(_env_mod, "is_rgb_env", None)
        if not callable(is_rgb_env_fn):
            return

        try:
            # In case the observation space is RGB, warn if MLP policy is used
            is_rgb = is_rgb_env_fn(self.train_env)
            policy = getattr(self.config, "policy", None) or ""
            is_mlp = isinstance(policy, str) and policy.lower() == "mlp"
            if is_rgb and is_mlp:
                print(
                    "Warning: Detected RGB image observations with MLP policy. "
                    "For pixel inputs, consider using CNN for better performance."
                )

            # In case the observation space is not RGB, warn if CNN policy is used
            is_cnn = isinstance(policy, str) and policy.lower() == "cnn"
            if not is_rgb and is_cnn:
                print(
                    "Warning: Detected non-RGB observations with CNN policy. "
                    "For non-image inputs, consider using MLP for better performance."
                )
        except Exception:
            # If env lacks expected attributes (e.g., test stubs), skip guidance
            return
    
    # ----- Evaluation scheduling helpers -----
    def _should_run_eval(self, epoch_idx: int) -> bool:
        """Return True if evaluation should run on this epoch index.

        Scheduling uses 1-based epoch numbers for human-friendly config:
        - Let E = epoch_idx + 1
        - If eval_freq_epochs is None or <= 0: never evaluate
        - If eval_warmup_epochs <= 0: evaluate at E == 1 and then on multiples of eval_freq_epochs
        - If eval_warmup_epochs > 0: skip all E <= eval_warmup_epochs, then evaluate on
          the configured cadence grid (i.e., E % eval_freq_epochs == 0). Example:
          warmup=50 and freq=15 -> first eval at E=60 (epoch_idx=59).
        """
        # If eval_freq_epochs is None or <= 0, never evaluate
        freq = self.config.eval_freq_epochs
        try:
            if freq is None or int(freq) <= 0:
                return False
        except Exception:
            return False

        warmup = int(self.config.eval_warmup_epochs or 0)
        E = int(epoch_idx) + 1
        if warmup <= 0:
            # First epoch (E==1) then multiples of freq
            return E == 1 or (E % int(freq) == 0)

        # With warmup, skip boundary and align to the cadence grid strictly after warmup
        if E <= warmup:
            return False

        return (E % int(freq)) == 0

    # -------------------------
    # Pre-prompt guidance helpers
    # -------------------------

    def _create_wandb_logger(self):
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        def _sanitize_name(name: str) -> str:
            return name.replace("/", "-").replace("\\", "-")

        # If a W&B run is already active (e.g., under a sweep), avoid re-initializing
        # and do not attempt to overwrite the run config. PL's WandbLogger will attach
        # to the existing run when present.
        try:
            import wandb  # type: ignore
            active_run = getattr(wandb, "run", None)
        except Exception:
            active_run = None

        if active_run is not None:
            wandb_logger = WandbLogger(log_model=True)
        else:
            project_name = self.config.project_id if self.config.project_id else _sanitize_name(self.config.env_id)
            experiment_name = f"{self.config.algo_id}-{self.config.seed}"
            wandb_logger = WandbLogger(
                project=project_name,
                name=experiment_name,
                log_model=True,
                config=asdict(self.config),
            )

        # Define the common step metric; works both for new and attached runs
        wandb_run = wandb_logger.experiment
        try:
            wandb_run.define_metric("*", step_metric="train/total_timesteps")
        except Exception:
            pass

        return wandb_logger
    
    def _build_trainer_callbacks(self):
        """Assemble trainer callbacks, with an optional end-of-training report."""
        # Lazy imports to avoid heavy deps at module import time
        from trainer_callbacks import (
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

        try:
            reward_threshold = self.train_env.get_reward_threshold()
        except Exception:
            reward_threshold = None
        earlystop_train_reward_cb = (
            EarlyStoppingCallback(
                "train/ep_rew_mean",
                reward_threshold,
                mode="max",
                verbose=False,
            )
            if (self.config.early_stop_on_train_threshold and reward_threshold is not None)
            else None
        )
        if earlystop_train_reward_cb:
            callbacks.append(earlystop_train_reward_cb)

        # Early stop when mean validation reward reaches a threshold
        try:
            reward_threshold = self.validation_env.get_reward_threshold()
        except Exception:
            reward_threshold = None
        earlystop_eval_reward_cb = (
            EarlyStoppingCallback(
                "eval/ep_rew_mean",
                reward_threshold,
                mode="max",
                verbose=False,
            )
            if (self.config.early_stop_on_eval_threshold and reward_threshold is not None)
            else None
        )
        if earlystop_eval_reward_cb:
            callbacks.append(earlystop_eval_reward_cb)

        # When training ends, write a report describing on the training went
        report_cb = EndOfTrainingReportCallback(
            filename="report.md"
        )
        callbacks.append(report_cb)

        return callbacks

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

    # Provide a scikit-like API where .fit() forwards to learn() for convenience
    # (used by tests and top-level scripts). Keep signature minimal.
    def fit(self):  # type: ignore[override]
        self.learn()

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

    def _get_training_progress(self):
        # Compute progress only when max_timesteps is configured; otherwise treat as 0.
        try:
            total = float(self.config.max_timesteps) if self.config.max_timesteps is not None else None
        except Exception:
            total = None
        if total is None or total <= 0.0:
            return 0.0
        total_steps = float(self.train_collector.total_steps)
        return max(0.0, min(total_steps / total, 1.0))

    def _update_schedules(self):
        self._update_schedules__policy_lr()

    # TODO: generalize scheduling support
    def _update_schedules__policy_lr(self):
        if self.config.policy_lr_schedule != "linear": return
        progress = self._get_training_progress()
        new_policy_lr = max(self.config.policy_lr * (1.0 - progress), 0.0)
        self._change_optimizers_policy_lr(new_policy_lr)
        # Log scheduled LR under train namespace
        self.log_metrics({"policy_lr": new_policy_lr}, prefix="train")

    def _change_optimizers_policy_lr(self, policy_lr):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups: pg["lr"] = policy_lr

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

        # Update best-so-far trackers when ep_rew_mean is present
        try:
            if isinstance(prefix, str) and "ep_rew_mean" in metrics:
                v = metrics.get("ep_rew_mean")
                if isinstance(v, (int, float)):
                    prev = self._best_rewards.get(prefix)
                    self._best_rewards[prefix] = v if prev is None else max(float(prev), float(v))
        except Exception:
            # Never break logging due to best tracking
            pass

        for k, v in prefixed.items():
            if k.endswith("action_dist"): continue
            if not isinstance(v, (int, float)): continue
            history = self._terminal_history.setdefault(k, [])
            step = self._last_step_for_terminal if k != "train/total_timesteps" else int(v)
            history.append((step, float(v)))

    # TODO: not sure about this 
    def _flush_metrics(self, *, log_to_lightning: bool = True):
        """
        Compute means from the metrics buffer and clear it.

        When log_to_lightning is True (default), forward the aggregated
        metrics to Lightning's logger via self.log_dict. Some lifecycle hooks
        (e.g., on_fit_end) disallow self.log(), so callers can set
        log_to_lightning=False to avoid Lightning logging in those contexts.
        """
        means = self._metrics_buffer.means()

        # Surface best-so-far episode rewards as explicit metrics
        try:
            bt = self._best_rewards.get("train")
            if bt is not None:
                means["train/ep_rew_best"] = float(bt)
        except Exception:
            pass
        try:
            be = self._best_rewards.get("eval")
            if be is not None:
                means["eval/ep_rew_best"] = float(be)
        except Exception:
            pass
        try:
            if log_to_lightning:
                # Forward aggregated metrics to Lightning when allowed
                self.log_dict(means)
        finally:
            # Always clear buffer regardless of logging outcome
            self._metrics_buffer.clear()
        return means

    # -------------------------
    # Small testable helpers
    # -------------------------
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Replace path separators with dashes for display/logging names."""
        return str(name).replace("/", "-").replace("\\", "-")

    @staticmethod
    def _compute_validation_controls(eval_freq_epochs):
        """Return dict with PL validation controls given an eval frequency.

        When eval_freq_epochs is None, validation is disabled.
        Otherwise, validation runs once per eval_freq_epochs.
        """
        # Treat None or non-positive values as disabled
        if eval_freq_epochs is None:
            return {"limit_val_batches": 0, "check_val_every_n_epoch": 1}
        try:
            if int(eval_freq_epochs) <= 0:
                return {"limit_val_batches": 0, "check_val_every_n_epoch": 1}
        except Exception:
            return {"limit_val_batches": 0, "check_val_every_n_epoch": 1}
        return {"limit_val_batches": 1.0, "check_val_every_n_epoch": int(eval_freq_epochs)}

        
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.policy_model.parameters(), 
            lr=self.config.policy_lr
        )

    # -------------------------
    # Gradient norm diagnostics
    # -------------------------
    def _compute_param_group_grad_norm(self, params):
        """Compute L2 norm of gradients for a parameter iterable.

        Ignores parameters with None gradients. Returns 0.0 if no grads present.
        """
        import math
        total_sq = 0.0
        has_grad = False
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            has_grad = True
            # Use .detach() to avoid graph tracking; flatten to 1D before norm
            total_sq += float(g.detach().data.norm(2).item() ** 2)
        if not has_grad:
            return 0.0
        return math.sqrt(total_sq)

    def _log_policy_grad_norms(self):
        """Log gradient norms for actor head, critic head, and shared trunk.

        Supports actor-critic models that expose `policy_head` and `value_head`.
        The shared trunk is computed as all parameters excluding head parameters
        (e.g., backbone/CNN feature extractor).
        """
        model = getattr(self, "policy_model", None)
        if model is None:
            return

        # Identify heads if present
        policy_head = getattr(model, "policy_head", None)
        value_head = getattr(model, "value_head", None)

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
        trunk_params = [p for p in model.parameters() if id(p) not in head_param_ids]

        metrics = {
            "grad_norm/actor_head": self._compute_param_group_grad_norm(actor_params),
            "grad_norm/critic_head": self._compute_param_group_grad_norm(critic_params) if value_head is not None else 0.0,
            "grad_norm/trunk": self._compute_param_group_grad_norm(trunk_params),
        }
        self.log_metrics(metrics, prefix="train")
