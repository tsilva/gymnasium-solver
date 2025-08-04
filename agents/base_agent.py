import os
import sys
import time
import torch
import wandb
import pytorch_lightning as pl
from utils.rollouts import RolloutCollector
from utils.misc import prefix_dict_keys, StdoutMetricsTable
from utils.video_logger_callback import VideoLoggerCallback
from utils.misc import create_dummy_dataloader
from utils.checkpoint import ModelCheckpointCallback

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()

        # Store core attributes
        self.config = config

        # Create environment builder
        from utils.environment import build_env

        # TODO: create this only for training? create on_fit_start() and destroy with on_fit_end()?
        self.train_env = build_env(
            config.env_id,
            n_envs=config.n_envs,
            seed=config.seed,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            obs_type=config.obs_type, # TODO: atari only?
        )
       
        self.eval_env = build_env(
            config.env_id,
            n_envs=1,
            seed=config.seed + 1000,  # Use a different seed for evaluation
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            frame_stack=config.frame_stack,
            obs_type=config.obs_type,
            render_mode="rgb_array",
            record_video=True,
            record_video_kwargs={
                "video_length": 100
            }
        )

        # Training state
        self.start_time = None # TODO: cleaner way of measuring this?
        self._n_updates = 0 # TODO: is this required?
        
        # Best model tracking (maintained for compatibility with checkpoint callback)
        self.best_eval_reward = float('-inf')
        self.best_model_path = None

        # Create the agent's models
        self.create_models()
        assert self.policy_model is not None, "Policy model must be created in create_models()"
    
        self.train_collector = RolloutCollector(
            self.train_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams()
        )

        self.eval_collector = RolloutCollector(
            self.eval_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams() # TODO: softcode
        )
    
    def create_models(self):
        raise NotImplementedError("Subclass must implement create_models()")
 
    def train_on_batch(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement train_on_batch()") # TODO: use override_required decorator
    
    def rollout_collector_hyperparams(self):
        return {
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda
        }

    def on_fit_start(self):
        self.start_time = time.time_ns()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
  
    def on_train_epoch_start(self):
        pass

    def train_dataloader(self):
        self._last_train_dataloader_epoch = self.current_epoch

        self.train_collector.collect()
        dataloader = self.train_collector.create_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        return dataloader
    
    def training_step(self, batch, batch_idx):
        # Assert that train_dataloader is called once per epoch
        # (this is the method that is performing the per-epoch rollout)
        assert self._last_train_dataloader_epoch == self.current_epoch, "train_dataloader must be called once per epoch"
        
        for _ in range(self.config.n_epochs): 
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

            self._n_updates += 1
            
    # TODO: aggregate logging
    def on_train_epoch_end(self):
        # Check for early stopping based on n_timesteps limit
        rollout_metrics = self.train_collector.get_metrics()
        total_timesteps = rollout_metrics["total_timesteps"]
        if self.config.n_timesteps is not None and total_timesteps >= self.config.n_timesteps:
            print(f"Stopping training at epoch {self.current_epoch} with {self.total_timesteps} timesteps >= limit {self.config.n_timesteps}")
            self.trainer.should_stop = True
        
        # Extract action distribution for histogram logging
        action_distribution = rollout_metrics.pop("action_distribution", None)

        time_metrics = self._get_time_metrics()
        metrics_dict = {
            "train/epoch": self.current_epoch, # TODO: is this the same value as in epoch_start?
            "train/n_updates": self._n_updates,
            **prefix_dict_keys(rollout_metrics, "rollout"),
            **prefix_dict_keys(time_metrics, "time")
        }
        
        # Log regular metrics
        self.log_dict(metrics_dict)
        
    def val_dataloader(self):
        return create_dummy_dataloader()

    def on_validation_epoch_start(self):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self._run_evaluation()
        
    def on_validation_epoch_end(self):
        pass

    def on_fit_end(self):
        time_elapsed = self._get_time_metrics()["time_elapsed"]
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")
        
        # Checkpoint completion summary is now handled by ModelCheckpointCallback

    def _process_eval_videos(self):
        """Process eval videos immediately to ensure they're logged at the correct timestep."""
        # Find the video logger callback
        video_logger = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, VideoLoggerCallback):
                video_logger = callback
                break
        
        if video_logger:
            # Process eval videos immediately
            video_logger._process(self.trainer, "eval")
    
    # TODO: should we change method name?
    def run_training(self):
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        # Convert config to dictionary for logging
        config_dict = asdict(self.config)
        
        # Sanitize project name for wandb (replace invalid characters)
        project_name = self.config.env_id.replace("/", "-").replace("\\", "-") # TODO: softcode this
        experiment_name = f"{self.config.algo_id}-{self.config.seed}" # TODO: softcode this        

        # Use regular WandbLogger
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=True,
            config=config_dict,
        )
        
        # Define wandb metrics to prevent step ordering issues
        if wandb_logger.experiment:
            # Define step-based metrics to ensure proper ordering
            wandb_logger.experiment.define_metric("train/*", step_metric="trainer/global_step")
            wandb_logger.experiment.define_metric("eval/*", step_metric="trainer/global_step")
            wandb_logger.experiment.define_metric("rollout/*", step_metric="trainer/global_step")
            wandb_logger.experiment.define_metric("time/*", step_metric="trainer/global_step")
        
        # Create video logging callback
        video_logger_cb = VideoLoggerCallback(
            media_root="videos",        # where you will drop files
            namespace_depth=1,          # "episodes" from train/episodes/ or eval/episodes/
            #log_interval_s=5.0,         # scan at most every 5 seconds
            #max_per_key=8,              # avoid spamming the panel
        )
        
        # Create checkpoint callback
        checkpoint_cb = ModelCheckpointCallback(
            checkpoint_dir=getattr(self.config, 'checkpoint_dir', 'checkpoints'),
            monitor="eval/ep_rew_mean",
            mode="max",
            save_last=True,
            save_threshold_reached=True,
            resume=getattr(self.config, 'resume', False)
        )
        
        # Create algorithm-specific metric rules from config
        from utils.config import get_metric_precision_dict, get_metric_delta_rules, get_algorithm_metric_rules
        metric_precision = get_metric_precision_dict()
        metric_delta_rules = get_metric_delta_rules()
        algo_metric_rules = get_algorithm_metric_rules(self.config.algo_id)
        
        # TODO: clean this up
        printer = StdoutMetricsTable(
            every_n_steps=200,   # print every 200 optimizer steps
            every_n_epochs=1,    # and at the end of every epoch
            digits=4,
            metric_precision=metric_precision,
            metric_delta_rules=metric_delta_rules,
            algorithm_metric_rules=algo_metric_rules  # Pass algorithm-specific rules
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            # TODO: softcode this
            log_every_n_steps=1, # TODO: softcode this
            max_epochs=self.config.max_epochs if self.config.max_epochs is not None else -1,
            enable_progress_bar=False,
            enable_checkpointing=False,  # Disable built-in checkpointing, use our custom callback
            accelerator="cpu",  # Use CPU for training # TODO: softcode this
            reload_dataloaders_every_n_epochs=1,#self.config.n_epochs
            check_val_every_n_epoch=self.config.eval_freq_epochs,  # Run validation every epoch
            callbacks=[printer, video_logger_cb, checkpoint_cb]  # Add checkpoint callback
        )
        trainer.fit(self)
    
    def _get_time_metrics(self):
        total_timesteps = self.train_collector.get_metrics()["total_timesteps"]
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = total_timesteps / time_elapsed
        return {
            "total_timesteps": total_timesteps,
            "time_elapsed": int(time_elapsed),
            "fps": int(fps)
        }
    
    # TODO: check how sb3 does eval_async
    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    def _run_evaluation(self):
        metrics = self._eval()
                
        # Extract action distribution for histogram logging
        action_distribution = metrics.pop("action_distribution", None)
        
        # Process videos immediately after evaluation, before logging metrics
        # This ensures videos and metrics are logged at the same timestep
        self._process_eval_videos()
        
        self.log_dict(prefix_dict_keys(metrics, "eval")) # TODO: overrrid log_dict and add prefixig support
        
        # Log action distribution as histogram to WandB for evaluation
        if action_distribution is not None and len(action_distribution) > 0:
            if hasattr(self.logger, 'experiment') and self.logger.experiment:
                self.logger.log_metrics({
                    "eval/action_distribution": wandb.Histogram(action_distribution)
                }, step=self.global_step)
    
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: consider making recording a rollout collector concern again (cleaner separation of concerns)
    def _eval(self):
        assert self.eval_env.num_envs == 1, "Evaluation should be run with a single environment instance"
        
        # Ensure output video directory exists
        assert wandb.run is not None, "wandb.init() must run before building the env"
        
        import random
        self.eval_collector.set_seed(random.randint(0, 1000000))  # Set a random seed for evaluation

        # Create video directory structure to match logger expectations
        video_root = os.path.join(wandb.run.dir, "videos", "eval", "episodes")
        os.makedirs(video_root, exist_ok=True)

        video_path = os.path.join(video_root, f"rollout_epoch_{self.current_epoch}.mp4")

        # TODO: make sure we can keep using same env across evals
        # Collect until we reach the required number of episodes
        self.eval_env.start_recording()
        metrics = self.eval_collector.get_metrics()
        total_episodes = metrics["total_episodes"]
        target_episodes = total_episodes + self.config.eval_episodes
        while total_episodes < target_episodes:
            self.eval_collector.collect(deterministic=self.config.eval_deterministic)
            metrics = self.eval_collector.get_metrics()
            total_episodes = metrics["total_episodes"]
        self.eval_env.stop_recording()
        self.eval_env.save_recording(video_path)

        return metrics

