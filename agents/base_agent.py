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

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()

        # Store core attributes
        self.config = config

        # Create environment builder
        from utils.environment import build_env
        self.build_env_fn = lambda seed, n_envs=config.n_envs, **kwargs: build_env(
            config.env_id,
            seed=seed,
            env_wrappers=config.env_wrappers,
            norm_obs=config.normalize_obs,
            n_envs=n_envs,
            frame_stack=config.frame_stack,
            obs_type=config.obs_type,
            render_mode="rgb_array",  # needed for video recording
            **kwargs
        )
        self.train_env = self.build_env_fn(config.seed)

        # TODO: these spec inspects should be centralized somewhere
        # Note: frame_stack is already applied in build_env, so observation_space.shape[0] 
        # already includes the frame stacking multiplication
        self.input_dim = self.train_env.observation_space.shape[0]
        self.output_dim = self.train_env.action_space.n
       
        # Training state
        self.start_time = None # TODO: cleaner way of measuring this?
        self.total_timesteps = 0
        self._n_updates = 0 # TODO: is this required?
        self._iterations = 0 # TODO: is this required?

        self.create_models()
        assert self.policy_model is not None, "Policy model must be created in create_models()"
    
    def create_models(self):
        raise NotImplementedError("Subclass must implement create_models()")
 
    def train_on_batch(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement train_on_batch()") # TODO: use override_required decorator
    
    def rollout_collector_hyperparams(self):
        return {
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda
        }
    
    def get_env_spec(self):
        """Get environment specification."""
        from utils.environment import get_env_spec
        env = self.build_env_fn(self.config.seed)
        return get_env_spec(env)

    def on_fit_start(self):
        self.start_time = time.time_ns()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # TODO: create this only for training? create on_fit_start() and destroy with on_fit_end()?
        self.train_collector = RolloutCollector(
            self.train_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams()
        )

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
        self._iterations += 1

        rollout_metrics = self.train_collector.get_metrics()
        self.total_timesteps = rollout_metrics["total_timesteps"]
        
        # Extract action distribution for histogram logging
        action_distribution = rollout_metrics.pop("action_distribution", None)

        time_metrics = self._get_time_metrics()
        metrics_dict = {
            # TODO: is this same as _iterations?
            "train/epoch": self.current_epoch, # TODO: is this the same value as in epoch_start?
            "train/n_updates": self._n_updates,
            "time/iterations": self._iterations,
            **prefix_dict_keys(rollout_metrics, "rollout"),
            **prefix_dict_keys(time_metrics, "time")
        }
        
        # Log regular metrics
        self.log_dict(metrics_dict)
        
        # Log action distribution as histogram to WandB
        if action_distribution is not None and len(action_distribution) > 0:
            if hasattr(self.logger, 'experiment') and self.logger.experiment:
                self.logger.experiment.log({
                    "rollout/action_distribution": wandb.Histogram(action_distribution)
                }, step=self.global_step) # TODO: use global step in videos too

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
        from tsilva_notebook_utils.colab import load_secrets_into_env # TODO: get rid of all references to this project
        from dataclasses import asdict
        from pytorch_lightning.loggers import WandbLogger

        # TODO: load this in main?
        _ = load_secrets_into_env(['WANDB_API_KEY'])
        
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
        
        # Create video logging callback
        video_logger = VideoLoggerCallback(
            media_root="videos",        # where you will drop files
            namespace_depth=1,          # "episodes" from train/episodes/ or eval/episodes/
            #log_interval_s=5.0,         # scan at most every 5 seconds
            #max_per_key=8,              # avoid spamming the panel
        )
        
        # TODO: clean this up
        printer = StdoutMetricsTable(
            every_n_steps=200,   # print every 200 optimizer steps
            every_n_epochs=1,    # and at the end of every epoch
            #include=[r"^train/", r"^val/", r"^time/", r"^rollout/", r"^eval/"],  # optional filters; remove to show everything
            # exclude=[r"^grad/"],           # example: drop noisy keys
            digits=4,
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            # TODO: softcode this
            log_every_n_steps=1, # TODO: softcode this
            max_epochs=self.config.max_epochs if self.config.max_epochs is not None else -1,
            enable_progress_bar=False,
            enable_checkpointing=False,  # Disable checkpointing for speed
            accelerator="cpu",  # Use CPU for training # TODO: softcode this
            reload_dataloaders_every_n_epochs=1,#self.config.n_epochs
            check_val_every_n_epoch=self.config.eval_freq_epochs,  # Run validation every epoch
            callbacks=[printer, video_logger]  # Add both callbacks
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
        env = self.build_env_fn(
            self.config.seed + 1000,
            n_envs=1,
            record_video=True,
            record_video_kwargs={
                "video_length": 100
            }
        )

        reward_threshold = env.get_reward_threshold()

        try: metrics = self._eval(env)
        finally: env.close()
        
        # Extract action distribution for histogram logging
        action_distribution = metrics.pop("action_distribution", None)
        
        # Process videos immediately after evaluation, before logging metrics
        # This ensures videos and metrics are logged at the same timestep
        self._process_eval_videos()
        
        self.log_dict(prefix_dict_keys(metrics, "eval")) # TODO: overrrid log_dict and add prefixig support
        
        # Log action distribution as histogram to WandB for evaluation
        if action_distribution is not None and len(action_distribution) > 0:
            if hasattr(self.logger, 'experiment') and self.logger.experiment:
                self.logger.experiment.log({
                    "eval/action_distribution": wandb.Histogram(action_distribution)
                }, step=self.global_step)

        # Check for early stopping based on reward threshold
        ep_rew_mean = metrics["ep_rew_mean"]
        
        if ep_rew_mean >= reward_threshold:
            print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {ep_rew_mean:.2f} >= threshold {reward_threshold}")
            self.trainer.should_stop = True
    
    # TODO: BUG: video first episode = 2, should be 1
    # TODO: add more stats to video (eg: episode, step, current reward, etc)
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: consider making recording a rollout collector concern again (cleaner separation of concerns)
    def _eval(self, env):
        assert env.num_envs == 1, "Evaluation should be run with a single environment instance"

        # Ensure output video directory exists
        assert wandb.run is not None, "wandb.init() must run before building the env"
        
        # Create video directory structure to match logger expectations
        video_root = os.path.join(wandb.run.dir, "videos", "eval", "episodes")
        os.makedirs(video_root, exist_ok=True)

        video_path = os.path.join(video_root, f"rollout_epoch_{self.current_epoch}.mp4")

        # TODO: make sure this is not calculating advantages
        collector = RolloutCollector(
            env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams() # TODO: softcode
        )

        # Collect until we reach the required number of episodes
        env.start_recording()
        total_episodes = 0
        while total_episodes < self.config.eval_episodes:
            collector.collect(deterministic=self.config.eval_deterministic)
            metrics = collector.get_metrics()
            total_episodes = metrics["total_episodes"]
        env.stop_recording()
        env.save_recording(video_path)

        return metrics

