import sys
import time
import json
import gymnasium
import pytorch_lightning as pl
from utils.rollouts import RolloutCollector
from utils.config import load_config
from utils.misc import prefix_dict_keys, print_namespaced_dict

# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, env_id: str):
        super().__init__()
        
        self.save_hyperparameters()

        self.env_id = env_id

        # TODO: these spec inspects should be centralized somewhere
        _env = gymnasium.make(env_id)
        self.input_dim = _env.observation_space.shape[0]
        self.output_dim = _env.action_space.n

        # Store core attributes
        algo_id = self.__class__.__name__.lower()
        config = load_config(env_id, algo_id)
        self.config = config

        # Create environment builder
        from utils.environment import build_env
        self.build_env_fn = lambda seed: build_env(
            config.env_id,
            seed=seed,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            n_envs=config.n_envs,
            vec_env_cls="DummyVecEnv"
        )

        # Training state
        self.start_time = None
        self._n_updates = 0
        self._epoch_metrics = {}
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
        # TODO: should I set this before training starts? think deeply to where this should be set
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(self.config.seed)

        self.start_time = time.time_ns()
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # TODO: create this only for training? create on_fit_start() and destroy with on_fit_end()?
        self.train_collector = RolloutCollector(
            self.build_env_fn(self.config.seed),
            self.policy_model,
            n_steps=self.config.n_steps,
            **self.config.rollout_collector_hyperparams()
        )

    def on_train_epoch_start(self):
        pass

    # TODO: assert this is being called every epoch
    def train_dataloader(self):
        self.train_collector.collect() # TODO: make this return dataloader with new dataset
        return self.train_collector.create_dataloader(batch_size=self.config.batch_size)
    
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
                # Clip grad norm
                max_grad_norm = 0.5
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_grad_norm)
                optimizer.step()

        self._n_updates += 1
        self.log_metrics({"train/n_updates": self._n_updates})

    def on_train_epoch_end(self):
        rollout_metrics = self.train_collector.get_metrics()
        self.log_metrics(prefix_dict_keys(rollout_metrics, "rollout"), prog_bar=["rollout/ep_rew_mean"], on_epoch=True) # TODO: move prefix inside

        time_metrics = self._get_time_metrics()
        self.log_metrics(prefix_dict_keys(time_metrics, "time"), on_epoch=True) # TODO: on_epoch?
        
        print_namespaced_dict(self._epoch_metrics)

        self._check_early_stop()

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
            enable_progress_bar=True,
            enable_checkpointing=False,  # Disable checkpointing for speed
            accelerator="cpu",  # Use CPU for training # TODO: softcode this
            reload_dataloaders_every_n_epochs=1#self.config.n_epochs
            #callbacks=[WandbCleanup()]
        )
        trainer.fit(self)

    def log_metrics(self, metrics, *, on_epoch=None, on_step=None, prog_bar=None):
        for key, value in metrics.items():
            _on_epoch = True if on_epoch is True or key in (on_epoch or []) else None
            _on_step =  True if on_step is True or key in (on_step or []) else None
            _prog_bar = True if prog_bar is True or key in (prog_bar or []) else None
            if _on_epoch is not False: self._epoch_metrics[key] = value
            self.log(key, value, on_epoch=_on_epoch, on_step=_on_step, prog_bar=_prog_bar)
    
    def _get_time_metrics(self):
        total_timesteps = self.train_collector.get_total_timesteps()
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = total_timesteps / time_elapsed
        return {
            "total_timesteps": total_timesteps,
            "time_elapsed": int(time_elapsed),
            "fps": int(fps)
        }
    
    def _check_early_stop(self):
        if not self.train_collector.is_reward_threshold_reached(): return
        ep_rew_mean = self.train_collector.get_ep_rew_mean()
        reward_threshold = self.train_collector.get_reward_threshold()
        print(f"Early stopping at epoch {self.current_epoch} with train mean reward {ep_rew_mean:.2f} >= threshold {reward_threshold}")
        self.trainer.should_stop = True

    # TODO: softcode this further
    def eval_and_render(self):
        # TODO: softcode this
        self.to("mps")
      
        from utils.environment import group_frames_by_episodes, render_episode_frames
        # TODO: make sure this is not calculating advantages
        eval_collector = RolloutCollector(
            self.build_env_fn(self.config.seed + 1000),  # Random seed for eval
            self.policy_model,
            n_episodes=self.config.eval_rollout_episodes,
            deterministic=True, # TODO: this is currently not implemented
            **self.config.rollout_collector_hyperparams()
        )
        try: eval_collector.collect(collect_frames=True) # TODO: is this collecting expected number of episodes? assert mean reward is not greater than allowed by env
        finally: del eval_collector
        print(json.dumps(eval_collector.get_metrics(), indent=2))        
        episode_frames = group_frames_by_episodes(eval_collector.trajectories)
        return render_episode_frames(episode_frames, out_dir="./tmp", grid=(2, 2), text_color=(0, 0, 0)) # TODO: review if eval collector should be deterministic or not
    