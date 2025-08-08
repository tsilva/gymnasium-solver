import os
import sys
import time
import torch
import wandb
import random
import pytorch_lightning as pl
from utils.environment import build_env
from utils.rollouts import RolloutCollector
from utils.misc import prefix_dict_keys
from utils.decorators import must_implement
from callbacks import PrintMetricsCallback, VideoLoggerCallback, ModelCheckpointCallback, HyperparameterScheduler
from torch.utils.data import DataLoader, TensorDataset

n_samples, sample_dim, batch_size = (1, 1, 1)
dummy_data = torch.zeros(n_samples, sample_dim)
dummy_target = torch.zeros(n_samples, sample_dim)
dataset = TensorDataset(dummy_data, dummy_target)
num_workers = os.cpu_count() - 1
validation_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()

        # Store core attributes
        self.config = config

        # Initialize throughput counters
        self.fit_start_time = None
        self.train_epoch_start_time = None
        self.train_epoch_start_timesteps = None
        self.validation_epoch_start_time = None
        self.validation_epoch_start_timesteps = None
        
        # Use subprocesses for parallel environments to improve performance,
        # but only if more than one environment is used.
        use_subproc = config.n_envs > 1

        # Create the environment that will be used for training
        self.train_env = build_env(
            config.env_id,
            seed=config.seed,
            n_envs=config.n_envs,
            subproc=use_subproc,
            obs_type=config.obs_type,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            render_mode=None,
            env_kwargs=config.env_kwargs
        )
       
        # Create the environment that will be used for evaluation
        # - Just 1 environment instance due to current rollout collector limitations
        # - Different seed to ensure different initial states
        # - Video recording enabled
        self.validation_env = build_env(
            config.env_id,
            seed=config.seed + 1000,
            n_envs=1,
            subproc=False,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            obs_type=config.obs_type,
            env_kwargs=config.env_kwargs,
            render_mode="rgb_array",
            record_video=True,
            record_video_kwargs={
                "video_length": 100 # TODO: softcode this
            }
        )

        # Create the models that the agent will require (eg: policy, value function, etc.)
        self.create_models()
        assert self.policy_model is not None, "Policy model must be created in create_models()"

        # Create the rollout collector that will be used to 
        # collect trajectories from the training environment
        self.train_collector = RolloutCollector(
            self.train_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams()
        )

        # Create the rollout collector that will be used to
        # collect trajectories from the validation environment
        self.validation_collector = RolloutCollector(
            self.validation_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams()
        )

        # Initialize a dictionary to store epoch metrics
        # - Used to collect metrics during training and validation epochs
        # - Metrics are flushed at the end of each epoch
        # - This avoids performance drops caused by using Lightning's logger
        #   which can be slow when logging many metrics
        # - Metrics are logged using self.log_metrics() method
        self._epoch_metrics = {}

        self._train_dataloader = None  # Placeholder for training dataloader
    
    @must_implement
    def create_models(self):
        pass
 
    @must_implement
    def train_on_batch(self, batch, batch_idx):
        pass
    
    def rollout_collector_hyperparams(self):
        return {
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda
        }

    def on_fit_start(self):
        self.fit_start_time = time.time_ns()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}") # TODO: use self.fit_start_time for logging

    def train_dataloader(self):
        # Initialize throughput calculation metrics
        self.train_epoch_start_time = time.time_ns()
        train_metrics = self.train_collector.get_metrics()
        self.train_epoch_start_timesteps = train_metrics["total_timesteps"]
        
        if self.current_epoch == 0 or self.current_epoch % self.config.n_epochs == 0:
            # Collect a rollout to train on this epoch
            self.train_collector.collect()

            # Create a dataloader to feed the rollout as shuffled mini-batches
            self._train_dataloader = self.train_collector.create_dataloader(
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        return self._train_dataloader
      
    def on_train_epoch_start(self):
        # Training epoch start callback, nothing to do here
        pass

    def training_step(self, batch, batch_idx):
        losses = self.train_on_batch(batch, batch_idx)
        optimizers = self.optimizers()
        if not type(losses) in (list, tuple): losses = [losses]
        if not type(optimizers) in (list, tuple): optimizers = [optimizers]
        for idx, optimizer in enumerate(optimizers):
            loss = losses[idx]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm) # TODO: review impact, figure out why
            optimizer.step()
    
    # TODO: aggregate logging
    def on_train_epoch_end(self):
        # Calculate FPS
        time_elapsed = max((time.time_ns() - self.train_epoch_start_time) / 1e9, sys.float_info.epsilon)
        rollout_metrics = self.train_collector.get_metrics()
        total_timesteps = rollout_metrics["total_timesteps"]
        timesteps_elapsed = total_timesteps - self.train_epoch_start_timesteps
        epoch_fps = int(timesteps_elapsed / time_elapsed)
        
        # TODO: temporary, remove this
       # if epoch_fps < 1000:
       #    print(f"Warning: Training FPS is low ({epoch_fps}). Consider reducing n_envs or n_steps to improve performance.")
       #     pass

        # TODO: softcode this
        rollout_metrics.pop("action_distribution")

        # Log metrics 
        self.log_metrics({
            **rollout_metrics,
            "epoch": self.current_epoch, # TODO: is this the same value as in epoch_start?
            "epoch_fps": epoch_fps,
        }, prefix="train")
        
        # In case we have reached the maximum number of training timesteps then stop training
        if self.config.n_timesteps is not None and total_timesteps >= self.config.n_timesteps:
            print(f"Stopping training at epoch {self.current_epoch} with {total_timesteps} timesteps >= limit {self.config.n_timesteps}")
            self.trainer.should_stop = True

        self._flush_metrics()

    def val_dataloader(self):
        return validation_dataloader

    def on_validation_epoch_start(self):
        assert self.current_epoch == 0 or (self.current_epoch + 1) % self.config.eval_freq_epochs == 0, f"Validation epoch {self.current_epoch} is not a multiple of eval_freq_epochs {self.config.eval_freq_epochs}"
        self.validation_epoch_start_time = time.time_ns()
        validation_metrics = self.validation_collector.get_metrics()
        self.validation_epoch_start_timesteps = validation_metrics["total_timesteps"]

    # TODO: check how sb3 does eval_async
    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: consider making recording a rollout collector concern again (cleaner separation of concerns)
    # TODO: consider using rollout_ep
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: currently support single environment evaluation
        assert self.validation_env.num_envs == 1, "Evaluation should be run with a single environment instance"

        # Collect until we reach the required number of episodes
        # NOTE: processing/saving video is a bottleneck that will make next training epoch be slower,
        # if you see train/fps drops, make video recording less frequent by adjusting `eval_recording_freq_epochs`
        record_video = self.current_epoch == 0 or (self.current_epoch + 1) % self.config.eval_recording_freq_epochs == 0
        
        # Use run-specific video directory if available, otherwise fallback to wandb.run.dir
        video_path = self.run_manager.get_video_dir() / "eval" / "episodes" / f"rollout_epoch_{self.current_epoch}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path = str(video_path)
   
        with self.validation_env.recorder(video_path, record_video=record_video): # TODO: make rew window = config.eval_episodes
            metrics = self.validation_collector.get_metrics()
            total_episodes = metrics["total_episodes"]
            target_episodes = total_episodes + self.config.eval_episodes
            while total_episodes < target_episodes:
                self.validation_collector.collect()#deterministic=self.config.eval_deterministic) # TODO: this still won't be as fast as possible because it will have run steps that will not be used 
                metrics = self.validation_collector.get_metrics()
                total_episodes = metrics["total_episodes"]

    def on_validation_epoch_end(self):
        assert self.current_epoch == 0 or (self.current_epoch + 1) % self.config.eval_freq_epochs == 0, f"Validation epoch {self.current_epoch} is not a multiple of eval_freq_epochs {self.config.eval_freq_epochs}"
       
        # Calculate FPS
        time_elapsed = max((time.time_ns() - self.validation_epoch_start_time) / 1e9, sys.float_info.epsilon)
        rollout_metrics = self.validation_collector.get_metrics()
        total_timesteps = rollout_metrics["total_timesteps"]
        timesteps_elapsed = total_timesteps - self.validation_epoch_start_timesteps
        epoch_fps = int(timesteps_elapsed / time_elapsed)

        # TODO: softcode this
        rollout_metrics.pop("action_distribution", None)  # Remove action distribution if present

        # Log metrics
        self.log_metrics({
            **rollout_metrics,
            "epoch": int(self.current_epoch),
            "epoch_fps": epoch_fps
        }, prefix="eval")

        self._flush_metrics()

    def on_fit_end(self):
        time_elapsed = max((time.time_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")
        
    # TODO: should we change method name?
    def _run_training(self):
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger
        from utils.run_manager import RunManager
        from utils.logging import capture_all_output, log_config_details

        # Create wandb logger and run manager
        project_name = self.config.project_id if self.config.project_id else self.config.env_id.replace("/", "-").replace("\\", "-")
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=True,
            config=asdict(self.config)
        )
        
        # Setup run directory management
        self.run_manager = RunManager()
        run_dir = self.run_manager.setup_run_directory(wandb_logger.experiment)
        run_logs_dir = str(self.run_manager.get_logs_dir())
        
        print(f"Run directory: {run_dir}")
        print(f"Run ID: {self.run_manager.run_id}")
        print(f"Logs will be saved to: {run_logs_dir}")
        
        # Set up comprehensive logging using run-specific logs directory
        with capture_all_output(config=self.config, log_dir=run_logs_dir):
            # Log configuration details
            log_config_details(self.config)
            
            # Save configuration to run directory
            config_path = self.run_manager.save_config(self.config)
            print(f"Configuration saved to: {config_path}")
            
            # Define step-based metrics to ensure proper ordering
            if wandb_logger.experiment:
                wandb_logger.experiment.define_metric("train/*", step_metric="train/total_timesteps")
                wandb_logger.experiment.define_metric("eval/*", step_metric="train/total_timesteps")
            
            # Create video logging callback using run-specific directory
            video_logger_cb = VideoLoggerCallback(
                media_root=str(self.run_manager.get_video_dir()),  # Use run-specific video directory
                namespace_depth=1,          # "episodes" from train/episodes/ or eval/episodes/
                #log_interval_s=5.0,         # scan at most every 5 seconds
                #max_per_key=8,              # avoid spamming the panel
            )
            
            # TODO: review early stopping logic
            # Create checkpoint callback using run-specific directory
            checkpoint_cb = ModelCheckpointCallback(
                checkpoint_dir=str(self.run_manager.get_checkpoint_dir()),
                monitor="eval/ep_rew_mean",
                mode="max",
                save_last=True,
                save_threshold_reached=True,
                resume=getattr(self.config, 'resume', False)
            ) if self.config.algo_id != "qlearning" else None
            
            # Create algorithm-specific metric rules from config
            from utils.config import get_metric_precision_dict, get_metric_delta_rules, get_algorithm_metric_rules
            metric_precision = get_metric_precision_dict()
            metric_delta_rules = get_metric_delta_rules()
            algo_metric_rules = get_algorithm_metric_rules(self.config.algo_id)
            
            # TODO: clean this up
            # TODO: print myself without printer callback?
            printer_cb = PrintMetricsCallback(
                # TODO: should this be same as log_every_n_steps?
                every_n_steps=200,   # print every 200 optimizer steps
                # TODO: not sure if this is working
                every_n_epochs=10,    # and at the end of every epoch
                digits=4, # TODO: is this still needed?
                # TODO: pass single metric config
                metric_precision=metric_precision,
                metric_delta_rules=metric_delta_rules,
                algorithm_metric_rules=algo_metric_rules  # Pass algorithm-specific rules
            )

            # Create hyperparameter scheduler callback
            hyperparam_cb = HyperparameterScheduler(
                control_dir=str(self.run_manager.run_dir / "hyperparam_control"),
                check_interval=2.0,  # Check every 2 seconds
                enable_lr_scheduling=True,
                enable_manual_control=True,
                verbose=True
            )

            callbacks = [x for x in [printer_cb, video_logger_cb, checkpoint_cb, hyperparam_cb] if x is not None]  # Filter out None callbacks

            trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=self.config.max_epochs if self.config.max_epochs is not None else -1,
                enable_progress_bar=False,
                enable_checkpointing=False,  # Disable built-in checkpointing, use our custom callback
                accelerator="cpu",  # Use CPU for training # TODO: softcode this
                reload_dataloaders_every_n_epochs=1,#self.config.n_epochs
                check_val_every_n_epoch=self.config.eval_freq_epochs,  # Run validation every epoch
                num_sanity_val_steps=0,
                callbacks=callbacks
            )
            trainer.fit(self)

    def log_metrics(self, metrics, *, prefix=None):
        """
        Lightning logger caused significant performance drops, as much as 2x slower train/fps.
        Using custom metric collection / flushing logic to avoid this issue.
        """
        
        _metrics = metrics
        if prefix: _metrics = prefix_dict_keys(metrics, prefix)
        for key, value in _metrics.items():
            _list = self._epoch_metrics.get(key, [])
            _list.append(value)
            self._epoch_metrics[key] = _list

    def _flush_metrics(self):
        means = {}
        for key, values in self._epoch_metrics.items():
            mean = sum(values) / len(values) if values else 0
            means[key] = mean
        self.log_dict(means)
        self._epoch_metrics.clear()  # Clear metrics after logging to prevent accumulation
