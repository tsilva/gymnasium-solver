import os
import sys
import time
import torch
import wandb
import random
import pytorch_lightning as pl
from utils.environment import build_env
from utils.rollouts import RolloutCollector
from utils.misc import prefix_dict_keys, create_dummy_dataloader
from callbacks import PrintMetricsCallback, VideoLoggerCallback, ModelCheckpointCallback

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()

        # Store core attributes
        self.config = config


        self.fit_start_time = None
        self.train_epoch_start_time = None
        self.train_epoch_start_timesteps = None
        self.validation_epoch_start_time = None
        self.validation_epoch_start_timesteps = None
        
        self._n_updates = 0 # TODO: is this required?

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
       
        self.validation_env = build_env(
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
                "video_length": 100 # TODO: softcode this
            }
        )
        
        # TODO: this should be in callback?
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

        self.validation_collector = RolloutCollector(
            self.validation_env,
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
        self.fit_start_time = time.time_ns()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}") # TODO: use self.fit_start_time for logging

    def train_dataloader(self):
        self.train_epoch_start_time = time.time_ns()
        train_metrics = self.train_collector.get_metrics()
        self.train_epoch_start_timesteps = train_metrics["total_timesteps"]

        self._last_train_dataloader_epoch = self.current_epoch

        self.train_collector.collect()
        dataloader = self.train_collector.create_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        return dataloader
      
    def on_train_epoch_start(self):
        pass

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
    
    # TODO: when I enable uncomment this, train/fps is 2x faster
    #def log_dict(*args, **kwargs):
    #    pass

    # TODO: aggregate logging
    def on_train_epoch_end(self):
        # Calculate FPS
        time_elapsed = max((time.time_ns() - self.train_epoch_start_time) / 1e9, sys.float_info.epsilon)
        rollout_metrics = self.train_collector.get_metrics()
        total_timesteps = rollout_metrics["total_timesteps"]
        timesteps_elapsed = total_timesteps - self.train_epoch_start_timesteps
        fps = int(timesteps_elapsed / time_elapsed)
        print(fps)

        # TODO: softcode this
        rollout_metrics.pop("action_distribution")

        # Log metrics 
        self.log_dict(prefix_dict_keys({
            **rollout_metrics,
            "epoch": self.current_epoch, # TODO: is this the same value as in epoch_start?
            "n_updates": self._n_updates,
            "fps": fps,
        }, "train"), on_epoch=True)
        
        # In case we have reached the maximum number of training timesteps then stop training
        if self.config.n_timesteps is not None and total_timesteps >= self.config.n_timesteps:
            print(f"Stopping training at epoch {self.current_epoch} with {total_timesteps} timesteps >= limit {self.config.n_timesteps}")
            self.trainer.should_stop = True

    def val_dataloader(self):
        return create_dummy_dataloader()

    def on_validation_epoch_start(self):
        self.validation_epoch_start_time = time.time_ns()
        validation_metrics = self.validation_collector.get_metrics()
        self.validation_epoch_start_timesteps = validation_metrics["total_timesteps"]

    # TODO: check how sb3 does eval_async
    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: consider making recording a rollout collector concern again (cleaner separation of concerns)
    # TODO: consider using rollout_ep
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: currently support single environment evaluation
        assert self.validation_env.num_envs == 1, "Evaluation should be run with a single environment instance"
        
        self.validation_collector.set_seed(random.randint(0, 1000000))  # Set a random seed for evaluation

        # Collect until we reach the required number of episodes
        video_path = os.path.join(wandb.run.dir, f"videos/eval/episodes/rollout_epoch_{self.current_epoch}.mp4")
        with self.validation_env.recorder(video_path):
            metrics = self.validation_collector.get_metrics()
            total_episodes = metrics["total_episodes"]
            target_episodes = total_episodes + self.config.eval_episodes
            while total_episodes < target_episodes:
                self.validation_collector.collect(deterministic=self.config.eval_deterministic)
                metrics = self.validation_collector.get_metrics()
                total_episodes = metrics["total_episodes"]

    def on_validation_epoch_end(self):
        # Calculate FPS
        time_elapsed = max((time.time_ns() - self.validation_epoch_start_time) / 1e9, sys.float_info.epsilon)
        rollout_metrics = self.validation_collector.get_metrics()
        total_timesteps = rollout_metrics["total_timesteps"]
        timesteps_elapsed = total_timesteps - self.validation_epoch_start_timesteps
        fps = int(timesteps_elapsed / time_elapsed)

        # TODO: softcode this
        rollout_metrics.pop("action_distribution", None)  # Remove action distribution if present

        # Log metrics
        self.log_dict(prefix_dict_keys({
            **rollout_metrics,
            "epoch": self.current_epoch,
            "fps": fps
        }, "eval"), on_epoch=True)

    def on_fit_end(self):
        time_elapsed = max((time.time_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")
        
    # TODO: should we change method name?
    def _run_training(self):
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        # Use regular WandbLogger
        project_name = self.config.env_id.replace("/", "-").replace("\\", "-")
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=True,
            config=asdict(self.config)
        )
        
        # Define step-based metrics to ensure proper ordering
        if wandb_logger.experiment:
            wandb_logger.experiment.define_metric("train/*", step_metric="trainer/global_step")
            wandb_logger.experiment.define_metric("eval/*", step_metric="trainer/global_step")
        
        # Create video logging callback
        video_logger_cb = VideoLoggerCallback(
            media_root="videos",        # where you will drop files
            namespace_depth=1,          # "episodes" from train/episodes/ or eval/episodes/
            #log_interval_s=5.0,         # scan at most every 5 seconds
            #max_per_key=8,              # avoid spamming the panel
        )
        
        # TODO: review early stopping logic
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
        printer_cb = PrintMetricsCallback(
            # TODO: should this be same as log_every_n_steps?
            every_n_steps=200,   # print every 200 optimizer steps
            every_n_epochs=1,    # and at the end of every epoch
            digits=4, # TODO: is this still needed?
            # TODO: pass single metric config
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
            # TODO: these slowdown FPS a bit, but not too much
            callbacks=[printer_cb, video_logger_cb, checkpoint_cb]  # Add checkpoint callback
        )
        trainer.fit(self)