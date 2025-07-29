import sys
import time
import json
import gymnasium
import pytorch_lightning as pl
from utils.rollouts import RolloutCollector
from utils.misc import prefix_dict_keys, print_namespaced_dict

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()

        self.config = config

        # TODO: these spec inspects should be centralized somewhere
        _env = gymnasium.make(config.env_id)
        base_input_dim = _env.observation_space.shape[0]
        # Account for frame stacking - multiply by number of frames
        self.input_dim = base_input_dim * config.frame_stack
        self.output_dim = _env.action_space.n

        # Store core attributes
        self.config = config

        # Create environment builder
        from utils.environment import build_env
        self.build_env_fn = lambda seed, n_envs=config.n_envs, record_video=False: build_env(
            config.env_id,
            seed=seed,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            n_envs=n_envs,
            vec_env_cls="DummyVecEnv",
            reward_shaping=config.reward_shaping,
            frame_stack=config.frame_stack,
            record_video=record_video
        )
        self.train_env = self.build_env_fn(config.seed)
        self.eval_env = self.build_env_fn(config.seed + 1000, record_video=True)

        # Training state
        self.start_time = None
        self._epoch_metrics = {}
        self._n_updates = 0
        self._iterations = 0

        # TODO: move this to on_fit_start()?
        self.policy_model = None
        self.create_models()
    
    def create_models(self):
        raise NotImplementedError("Subclass must implement create_models()")
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement training_step()")

    def train_on_batch(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement train_on_batch()")
    
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
            **self.config.rollout_collector_hyperparams()
        )

    def on_train_epoch_start(self):
        pass

    # TODO: assert this is being called every epoch
    def train_dataloader(self):
        self.train_collector.collect()
        dataloader = self.train_collector.create_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        return dataloader
    
    def training_step(self, batch, batch_idx):
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
        time_metrics = self._get_time_metrics()
        self.log_metrics({
            "train/n_updates": self._n_updates,
            "time/iterations": self._iterations,
            **prefix_dict_keys(rollout_metrics, "rollout"),
            **prefix_dict_keys(time_metrics, "time")
        })

        self._flush_metrics()
        self._check_early_stop()
    
    def log_metrics(self, metrics):
        for key, value in metrics.items(): self._epoch_metrics[key] = value # TODO: assert no overwriting of keys
    
    # NOTE: beware of any changes to logging as torch lightning logging can slowdown training (expected performance is >6000 FPS on Macbook Pro M1)
    def _flush_metrics(self):
        # TODO: self.log is a gigantic bottleneck, currently halving performance;
        # ; must fix this bug while retaining FPS
        self.log_dict(self._epoch_metrics)
        if self.current_epoch % 10 == 0: print_namespaced_dict(self._epoch_metrics) # TODO: softcode this
        self._epoch_metrics = {}

    def on_fit_end(self):
        time_elapsed = self._get_time_metrics()["time_elapsed"]
        print(f"Training completed in {time_elapsed:.2f} seconds ({time_elapsed/60:.2f} minutes)")
        del self.train_collector

    # TODO: when does this run? should I run eval_rollout_collector here?
    #def validation_step(self, *args, **kwargs):
    #    return super().validation_step(*args, **kwargs)

    def train(self):
        from pytorch_lightning.loggers import WandbLogger
        from tsilva_notebook_utils.colab import load_secrets_into_env # TODO: get rid of all references to this project

        _ = load_secrets_into_env(['WANDB_API_KEY'])
        wandb_logger = WandbLogger(
            project=self.config.env_id,
            name=f"{self.config.algo_id}-{self.config.seed}",
            log_model=True
        )
        
        trainer = pl.Trainer(
            logger=wandb_logger,
            # TODO: softcode this
            log_every_n_steps=1, # TODO: softcode this
            max_epochs=self.config.max_epochs if self.config.max_epochs is not None else -1,
            enable_progress_bar=False,
            enable_checkpointing=False,  # Disable checkpointing for speed
            accelerator="cpu",  # Use CPU for training # TODO: softcode this
            reload_dataloaders_every_n_epochs=1#self.config.n_epochs
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
    
    def _check_early_stop(self):
        if self._iterations % 100 != 0: return # TODO: softcode frequency
        info = self.eval()
        reward_threshold = self.get_reward_threshold()
        ep_rew_mean = info["ep_rew_mean"]
        if ep_rew_mean < reward_threshold: return
        print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {ep_rew_mean:.2f} >= threshold {reward_threshold}")
        self.trainer.should_stop = True

    # TODO: consider recording single video with all episodes in sequence
    def eval(self):
        # TODO: currently only supports single environment evaluation because we need to 
        # be able to calculate mean episode reward using only completed episodes
        # TODO: if we add more envs we also need to make sure that each env collects N episodes, because if 
        # one env collects more episodes than others, the results will be biased

        # TODO: add create data loader method
        # TODO: make sure this is not calculating advantages
        # TODO: make sure policy is executed deterministically
        # TODO: render to video
        collector = RolloutCollector(
            self.eval_env,
            self.policy_model, # TODO: add eval freq (by steps)
            n_steps=self.config.n_steps,
            **self.config.rollout_collector_hyperparams() # TODO: do we need to pass this?
        )
        n_episodes = self.eval_env.num_envs * 2 # TODO: softcode this
        info = collector.collect_episodes(n_episodes)
        print(json.dumps(info, indent=2))        
        return info
       