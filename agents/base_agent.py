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
from utils.samplers import MultiPassRandomSampler

n_samples, sample_dim, batch_size = (1, 1, 1)
dummy_data = torch.zeros(n_samples, sample_dim)
dummy_target = torch.zeros(n_samples, sample_dim)
dataset = TensorDataset(dummy_data, dummy_target)
validation_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

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

        # Create the environment that will be used for training
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
                "video_length": 100,  # TODO: softcode this
            },
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
            **self.rollout_collector_hyperparams(),
        )

        # Create the rollout collector that will be used to
        # collect trajectories from the validation environment
        self.validation_collector = RolloutCollector(
            self.validation_env,
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.rollout_collector_hyperparams(),
        )

        # Initialize a dictionary to store epoch metrics
        # - Used to collect metrics during training and validation epochs
        # - Metrics are flushed at the end of each epoch
        # - This avoids performance drops caused by using Lightning's logger
        #   which can be slow when logging many metrics
        # - Metrics are logged using self.log_metrics() method
        self._epoch_metrics = {}

        # Internal holder to persist and reuse the train dataloader across epochs
        self._train_dataloader = None

    @must_implement
    def create_models(self):
        pass
 
    @must_implement
    def losses_for_batch(self, batch, batch_idx):
        pass
    
    def rollout_collector_hyperparams(self):
        return {
            **self.config.rollout_collector_hyperparams(),
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda
        }

    def on_fit_start(self):
        self.fit_start_time = time.time_ns()
        self.trajectories = self.train_collector.collect()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}") # TODO: use self.fit_start_time for logging

    def train_dataloader(self):
        assert self.current_epoch == 0, "train_dataloader should only be called once at the start of training"

        # Create a stable generator for reproducible shuffles if a seed is provided
        gen = torch.Generator()
        if getattr(self.config, 'seed', None) is not None:
            gen.manual_seed(int(self.config.seed))

        data_len = len(self.trajectories.observations)
        sampler = MultiPassRandomSampler(
            data_len=data_len,
            num_passes=self.config.n_epochs, generator=gen
        )

        class _IndexDataset(torch.utils.data.Dataset):
            def __init__(self, length: int):
                self._len = length
            def __len__(self) -> int:
                return self._len
            def __getitem__(self, idx: int) -> int:
                return idx

        from utils.rollouts import RolloutTrajectory
        def _collate_fn(idxs: list[int]):
            return RolloutTrajectory(
                observations=self.trajectories.observations[idxs],
                actions=self.trajectories.actions[idxs],
                rewards=self.trajectories.rewards[idxs],
                dones=self.trajectories.dones[idxs],
                old_log_prob=self.trajectories.old_log_prob[idxs], # TODO: change names
                old_values=self.trajectories.old_values[idxs],
                advantages=self.trajectories.advantages[idxs],
                returns=self.trajectories.returns[idxs],
                next_observations=self.trajectories.next_observations[idxs]
            )

        index_ds = _IndexDataset(data_len)
        return DataLoader(
            dataset=index_ds,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=_collate_fn,
            shuffle=False,  # MultiPassRandomSampler controls ordering
            num_workers=0,
            pin_memory=False,  # Pinning memory is not needed for CPU training
            persistent_workers=False
        )
      
    def on_train_epoch_start(self):
        self.epoch_time = time.time_ns()
        self.trajectories = self.train_collector.collect()

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

        total_timesteps = rollout_metrics["total_timesteps"]
        time_elapsed = max((time.time_ns() - self.fit_start_time) / 1e9, sys.float_info.epsilon)
        fps = total_timesteps / time_elapsed

        # Include recomputed learning rate and clip range in epoch metrics
        self.log_metrics({
            **rollout_metrics,
            "time_elapsed": time_elapsed,
            "epoch": self.current_epoch, # TODO: is this the same value as in epoch_start?
            "fps" : fps
        }, prefix="train")
        
        self._flush_metrics()
        
        self._update_schedules()

        # In case we have reached the maximum number of training timesteps then stop training
        if self.config.n_timesteps is not None and total_timesteps >= self.config.n_timesteps:
            print(f"Stopping training at epoch {self.current_epoch} with {total_timesteps} timesteps >= limit {self.config.n_timesteps}")
            self.trainer.should_stop = True
    
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
                self.validation_collector.collect(deterministic=self.config.eval_deterministic) # TODO: this still won't be as fast as possible because it will have run steps that will not be used 
                metrics = self.validation_collector.get_metrics()
                total_episodes = metrics["total_episodes"]

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

    def on_validation_epoch_end(self):
        # Validation epoch end is called after all validation steps are done
        # (nothing to do because we already did everything in the validation step)
        pass

    def on_fit_end(self):
        # Log training completion time
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
                enable_lr_scheduling=False,
                enable_manual_control=True,
                verbose=True
            )

            callbacks = [x for x in [printer_cb, video_logger_cb, checkpoint_cb, hyperparam_cb] if x is not None]  # Filter out None callbacks

            # Determine validation controls based on configuration
            eval_freq = getattr(self.config, 'eval_freq_epochs', None)
            limit_val_batches = 0 if eval_freq is None else 1.0  # disable validation entirely if None
            check_val_every_n_epoch = eval_freq if eval_freq is not None else 1  # value won't matter when limit_val_batches=0

            trainer = pl.Trainer(
                logger=wandb_logger,
                max_epochs=self.config.max_epochs if self.config.max_epochs is not None else -1,
                enable_progress_bar=False,
                enable_checkpointing=False,  # Disable built-in checkpointing, use our custom callback
                accelerator="cpu",  # Use CPU for training # TODO: softcode this
                # Reuse dataloaders across epochs to avoid worker respawn and construction overhead
                reload_dataloaders_every_n_epochs=0,
                val_check_interval=None,  # Disable built-in validation check interval
                check_val_every_n_epoch=check_val_every_n_epoch,
                limit_val_batches=limit_val_batches,
                num_sanity_val_steps=0,
                callbacks=callbacks
            )
            trainer.fit(self)

    def _backpropagate_and_step(self, losses):
        optimizers = self.optimizers()
        if not type(losses) in (list, tuple): losses = [losses]
        if not type(optimizers) in (list, tuple): optimizers = [optimizers]
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
        if self.config.learning_rate_schedule != 'linear': return
        progress = self._get_training_progress()
        new_learning_rate = max(self.config.learning_rate * (1.0 - progress), 0.0)
        self._change_optimizers_learning_rate(new_learning_rate)
        
        # TODO: this should not be logged here
        self.log_metrics({
            'learning_rate': new_learning_rate
        }, prefix="train")
    
    def _change_optimizers_learning_rate(self, learning_rate):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)): optimizers = [optimizers]
        for opt in optimizers:
            for pg in opt.param_groups:
                pg['lr'] = learning_rate

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
