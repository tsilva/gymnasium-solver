import os
import sys
import time
import wandb
import pytorch_lightning as pl
from utils.rollouts import RolloutCollector
from utils.misc import prefix_dict_keys, print_namespaced_dict
from utils.wandb import WandbLoggerAutomedia

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
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            n_envs=n_envs,
            vec_env_cls="DummyVecEnv",
            reward_shaping=config.reward_shaping, # TODO: generalize to specifying wrappers
            frame_stack=config.frame_stack,
            obs_type=config.obs_type,
            **kwargs
        )
        self.train_env = self.build_env_fn(config.seed)

        # TODO: these spec inspects should be centralized somewhere
        # Note: frame_stack is already applied in build_env, so observation_space.shape[0] 
        # already includes the frame stacking multiplication
        self.input_dim = self.train_env.observation_space.shape[0]
        self.output_dim = self.train_env.action_space.n
       
        # Training state
        self.start_time = None
        self.total_timesteps = 0
        self._epoch_metrics = {}
        self._n_updates = 0
        self._iterations = 0

        # TODO: move this to on_fit_start()?
        self.policy_model = None
        self.create_models()
    
    def create_models(self):
        raise NotImplementedError("Subclass must implement create_models()")
 
    def train_on_batch(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement train_on_batch()")
    
    def get_env_spec(self):
        """Get environment specification."""
        from utils.environment import get_env_spec
        env = self.build_env_fn(self.config.seed)
        return get_env_spec(env)
    
    def rollout_collector_hyperparams(self):
        """Get hyperparameters for the rollout collector. Can be overridden by subclasses."""
        return self.config.rollout_collector_hyperparams()

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
        
        import torch
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

        time_metrics = self._get_time_metrics()
        self.log_metrics({
            "train/n_updates": self._n_updates,
            "time/iterations": self._iterations,
            **prefix_dict_keys(rollout_metrics, "rollout"),
            **prefix_dict_keys(time_metrics, "time")
        })

        self._flush_metrics()
        #self._check_early_stop()

    def val_dataloader(self):
        """
        Return a dummy validation dataloader to trigger validation methods.
        In RL, we don't use traditional validation data but run evaluation episodes.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create a dummy dataset with a single item to trigger validation_step once per epoch
        dummy_data = torch.zeros(1, 1)
        dummy_target = torch.zeros(1, 1)
        dataset = TensorDataset(dummy_data, dummy_target)
        return DataLoader(dataset, batch_size=1)

    def on_validation_epoch_start(self):
        """Called at the start of the validation epoch."""
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Operates on a single batch of data from the validation set.
        For RL, this runs evaluation rollouts instead of processing validation batches.
        
        Args:
            batch: The output of your data iterable (dummy data for RL)
            batch_idx: The index of this batch
            dataloader_idx: The index of the dataloader (default 0)
            
        Returns:
            Dict with validation metrics or None
        """
        # Run evaluation episodes
        eval_metrics = self.run_evaluation()
        
        self.log_metrics(prefix_dict_keys(eval_metrics, "eval"))

        # Check for early stopping based on reward threshold
        reward_threshold = self.get_reward_threshold()
        ep_rew_mean = eval_metrics["ep_rew_mean"]
        
        if ep_rew_mean >= reward_threshold:
            print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {ep_rew_mean:.2f} >= threshold {reward_threshold}")
            self.trainer.should_stop = True
        
    def on_validation_epoch_end(self):
        self._flush_metrics()

    def log_metrics(self, metrics):
        for key, value in metrics.items(): self._epoch_metrics[key] = value # TODO: assert no overwriting of keys
    
    # NOTE: beware of any changes to logging as torch lightning logging can slowdown training (expected performance is >6000 FPS on Macbook Pro M1)
    def _flush_metrics(self, safe_logging=True):
        # TODO: self.log is a gigantic bottleneck, currently halving performance;
        # ; must fix this bug while retaining FPS
        # Capture the current step before logging for video alignment
        current_step = self.global_step if hasattr(self, 'global_step') else None
        
        if safe_logging:
            try:
                self.log_dict(self._epoch_metrics) # TODO: when does this flush?
            except Exception as e:
                # If we can't log through Lightning, log directly to wandb
                print(f"Warning: Could not log through Lightning ({e}), logging directly to wandb")
                if wandb.run is not None:
                    wandb.log(self._epoch_metrics, step=current_step)
        else:
            # Log directly to wandb when Lightning logging is not safe
            if wandb.run is not None:
                wandb.log(self._epoch_metrics, step=current_step)
                
        print_namespaced_dict(self._epoch_metrics)
        print(wandb.run.get_url())
        
        # Store the step used for metrics so videos can use the same step
        self._last_logged_step = current_step
        self._epoch_metrics = {}

    def on_fit_end(self):
        time_elapsed = self._get_time_metrics()["time_elapsed"]
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")
        print_namespaced_dict(self._epoch_metrics)

    def run_training(self):
        from pytorch_lightning.loggers import WandbLogger
        from tsilva_notebook_utils.colab import load_secrets_into_env # TODO: get rid of all references to this project
        from dataclasses import asdict

        _ = load_secrets_into_env(['WANDB_API_KEY'])
        
        # Convert config to dictionary for logging
        config_dict = asdict(self.config)
        
        # Sanitize project name for wandb (replace invalid characters)
        project_name = self.config.env_id.replace("/", "-").replace("\\", "-") # TODO: softcode this
        experiment_name = f"{self.config.algo_id}-{self.config.seed}" # TODO: softcode this        
        wandb_logger = WandbLoggerAutomedia(
            project=project_name,
            name=experiment_name,
            log_model=True,
            config=config_dict,
            #
            media_root="videos",        # where you will drop files
            namespace_depth=2,          # "phase/name" from path
            log_interval_s=5.0,         # scan at most every 5 seconds
            max_per_key=8,              # avoid spamming the panel
            commit=False                # don't change Lightning's step handling
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
            #callbacks=[WandbCleanup()]
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
    
    # TODO: softcode this
    def get_reward_threshold(self):
        from utils.environment import get_env_reward_threshold
        reward_threshold = get_env_reward_threshold(self.train_env)
        return reward_threshold
    
    # TODO: should eval freq be based on n_updates?
    def _check_early_stop(self):
        """
        Early stopping logic has been moved to validation_step().
        This method is kept for backward compatibility but doesn't do anything.
        """
        pass

    # TODO: consider recording single video with all episodes in sequence
    # TODO: consider moving to validation_step()
    # TODO: check how sb3 does eval_async
    # TODO: add more stats to video (eg: episode, step, current reward, etc)
    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    def run_evaluation(self):
        # Ensure output video directory exists
        assert wandb.run is not None, "wandb.init() must run before building the env"
        root = os.path.join(wandb.run.dir, "videos", "eval", "episodes") # TODO: ensure two directories because last two are the key
        os.makedirs(root, exist_ok=True)

        eval_seed = self.config.seed + 1000  # Use a different seed for evaluation
        eval_env = self.build_env_fn(
            eval_seed,
            n_envs=1,
            record_video=True,
            record_video_kwargs={
                "video_folder": root,
                "name_prefix": f"{int(time.time())}",
            }
        )
        try: return self._eval(eval_env)
        finally: eval_env.close()
    
    def _eval(self, env):
        # TODO: make sure this is not calculating advantages
        collector = RolloutCollector(
            env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams()
        )

        # Consider the eval episodes in the config to be 
        # the minimum required and find a multiple of 
        # num_envs that is above that minimum    
        eval_episodes = 0
        while eval_episodes < self.config.eval_episodes: eval_episodes += env.num_envs

        # TODO: review collect_episodes internals
        info = collector.collect_episodes(
            eval_episodes, 
            deterministic=self.config.eval_deterministic
        )

        #eval_metrics = prefix_dict_keys(info, "eval")
       # self.log_metrics(eval_metrics)
        
        # Flush eval metrics immediately to ensure video step alignment
        # Use safe_logging=False since we might be in a validation hook
        #self._flush_metrics(safe_logging=False)
        
        # Force immediate video scan and log at the same step as eval metrics
        # Use the exact step that was used for logging the metrics
        #if hasattr(self.logger, '_scan_and_log_once'):
        #    step_for_videos = getattr(self, '_last_logged_step', None)
        #    self.logger._scan_and_log_once(step=step_for_videos)
 
        return info
