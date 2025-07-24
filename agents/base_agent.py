import time
import json
import random
from dataclasses import  asdict
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from utils.rollouts import SyncRolloutCollector
from utils.config import load_config

# TODO: add test script for collection using dataset/dataloader
# TODO: should dataloader move to gpu?
# TODO: would converting trajectories to tuples in advance be faster?
class RolloutDataset(TorchDataset):
    def __init__(self):
        self.trajectories = None 

    def update(self, *trajectories):
        self.trajectories = trajectories

    def __len__(self):
        length = len(self.trajectories[0])
        return length

    def __getitem__(self, idx):
        item = tuple(t[idx] for t in self.trajectories)
        return item
    
# TODO: don't create these before lightning module ships models to device, otherwise we will collect rollouts on CPU
class BaseAgent(pl.LightningModule):
    
    def __init__(self, env_id: str, *, n_envs = "auto"):
        super().__init__()
        
        self.save_hyperparameters()

        # Store core attributes
        algo_id = self.__class__.__name__.lower()
        config = load_config(env_id, algo_id)
        self.config = config

        print(json.dumps(asdict(config), indent=2))

        # Create environment builder
        from utils.environment import build_env
        self.build_env_fn = lambda seed: build_env(
            config.env_id,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            n_envs=n_envs,
            seed=seed
        )

        # Training state
        self.automatic_optimization = False
        self.training_start_time = None
        self.total_steps = 0  # Track total training steps consumed

        # TODO: move this to on_fit_start()?
        self.policy_model = None
        self.value_model = None
        self.create_models()
        
    def forward(self, x):
        return self.policy_model(x)
    
    def create_models(self):
        """Override in subclass to create algorithm-specific models"""
        raise NotImplementedError("Subclass must implement create_models()")
    
    def compute_loss(self, batch):
        """Override in subclass to compute algorithm-specific loss"""
        raise NotImplementedError("Subclass must implement compute_loss()")
        
    def optimize_models(self, loss_results):
        """Override in subclass to implement algorithm-specific optimization"""
        raise NotImplementedError("Subclass must implement optimize_models()")

    def on_fit_start(self):
        # TODO: should I set this before training starts? think deeply to where this should be set
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(self.config.seed)

        self.training_start_time = time.time()
        self.total_steps = 0  # Reset step counter
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Common RL components
        # TODO: create this only for training? create on_fit_start() and destroy with on_fit_end()?
        self.train_rollout_dataset = RolloutDataset()
        self.train_rollout_collector = SyncRolloutCollector(
            lambda: self.build_env_fn(self.config.seed),
            self.policy_model,
            value_model=self.value_model,
            n_steps=self.config.train_rollout_steps,
            **self.config.rollout_collector_hyperparams()
        )
        trajectories, stats = self.train_rollout_collector.collect()
        self.train_rollout_dataset.update(*trajectories)
        self._train_rollout_stats = stats # TODO: do I need to do this

        # TODO: memory leaks? what if I initialize the dataloader with correct rollout size?
        # TODO: does this approach work where we swap the data underneath the dataloader between epochs?
        # TODO: only try enabling num workers after PPO converges
        import multiprocessing
        self.dataloader = DataLoader(
            self.train_rollout_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            # TODO: fix num_workers, training not converging when they are on
            # Pin memory is not supported on MPS
            #pin_memory=True if self.device.type != 'mps' else False,
            # TODO: Persistent workers + num_workers is fast but doesn't converge
            #persistent_workers=True,# if self.device.type != 'mps' else False,
            # Using multiple workers stalls the start of each epoch when persistent workers are disabled
            #num_workers=multiprocessing.cpu_count() -1#// 2 if self.device.type != 'mps' else 0
        )
    
    def on_train_epoch_start(self):
        pass

    def train_dataloader(self):
        return self.dataloader

    def training_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        self.total_steps += batch_size

        loss_results = self.compute_loss(batch)

        self._train_rollout_stats = {**(self._train_rollout_stats or {}), **loss_results}

        self.optimize_models(loss_results)

    def on_train_epoch_end(self):
        if self._train_rollout_stats:
            print(self._train_rollout_stats)
            mean_ep_reward = self._train_rollout_stats["mean_ep_reward"]
            self.log_metrics(self._train_rollout_stats, prog_bar=["mean_ep_reward"], prefix="train")
            self._train_rollout_stats = None  # Reset stats after logging

            if self.config.train_reward_threshold and mean_ep_reward >= self.config.train_reward_threshold:
                print(f"Early stopping at epoch {self.current_epoch} with train mean reward {mean_ep_reward:.2f} >= threshold {self.config.train_reward_threshold}")
                self.trainer.should_stop = True

        trajectories, stats = self.train_rollout_collector.collect()
        self.train_rollout_dataset.update(*trajectories)
        self._train_rollout_stats = stats # TODO: do I need to do this

        if (self.current_epoch + 1) % self.config.eval_rollout_interval == 0: 
            # TODO: reuse this with eval()
            eval_rollout_collector = SyncRolloutCollector(
                lambda: self.build_env_fn(random.randint(0, 10000)),  # Random seed for eval
                self.policy_model,
                n_episodes=self.config.eval_rollout_episodes,
                deterministic=True,
                **self.config.rollout_collector_hyperparams()
            )
            _, stats = eval_rollout_collector.collect() # TODO: is this collecting expected number of episodes? assert mean reward is not greater than allowed by env
            del eval_rollout_collector  # Clean up collector

            self.log_metrics(stats, prog_bar=["mean_ep_reward"], prefix="eval")
            
            mean_ep_reward = stats['mean_ep_reward']
            if self.config.eval_reward_threshold and mean_ep_reward >= self.config.eval_reward_threshold:
                print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {mean_ep_reward:.2f} >= threshold {self.config.eval_reward_threshold}")
                self.trainer.should_stop = True
    
    def on_fit_end(self):
        total_time = time.time() - self.training_start_time
        print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

        # TODO: are these set to None?
        del self.train_rollout_collector  # Clean up collector
        del self.train_rollout_dataset  # Clean up dataset

    # TODO: when does this run? should I run eval_rollout_collector here?
    #def validation_step(self, *args, **kwargs):
    #    return super().validation_step(*args, **kwargs)
    
    def log_metrics(self, metrics, *, on_epoch=None, on_step=None, prog_bar=None, prefix=None):
        for key, value in metrics.items():
            _on_epoch = True if on_epoch is True or key in (on_epoch or []) else None
            _on_step =  True if on_step is True or key in (on_step or []) else None
            _prog_bar = True if prog_bar is True or key in (prog_bar or []) else None
            _key = f"{prefix}/{key}" if prefix else key
            self.log(_key, value, on_epoch=_on_epoch, on_step=_on_step, prog_bar=_prog_bar)

    def train(self):
        from pytorch_lightning.loggers import WandbLogger
        from tsilva_notebook_utils.colab import load_secrets_into_env

        _ = load_secrets_into_env(['WANDB_API_KEY'])
        wandb_logger = WandbLogger(
            project=self.config.env_id,
            name=f"{self.config.algo_id}-{self.config.seed}",
            log_model=True
        )
        
        trainer = pl.Trainer(
            logger=wandb_logger,
            log_every_n_steps=1, # TODO: softcode this
            max_epochs=self.config.max_epochs,
            enable_progress_bar=True,
            enable_checkpointing=False,  # Disable checkpointing for speed
            accelerator="auto",
            reload_dataloaders_every_n_epochs=self.config.train_rollout_interval
            #callbacks=[WandbCleanup()]
        )
        trainer.fit(self)

    # TODO: softcode this further
    def eval_and_render(self):
        # TODO: softcode this
        self.to("mps")
      
        from utils.environment import group_frames_by_episodes, render_episode_frames
        # TODO: make sure this is not calculating advantages
        eval_rollout_collector = SyncRolloutCollector(
            lambda: self.build_env_fn(random.randint(0, 10000)),  # Random seed for eval
            self.policy_model,
            n_episodes=self.config.eval_rollout_episodes,
            deterministic=True,
            **self.config.rollout_collector_hyperparams()
        )
        try: trajectories, stats = eval_rollout_collector.collect(collect_frames=True) # TODO: is this collecting expected number of episodes? assert mean reward is not greater than allowed by env
        finally: del eval_rollout_collector
        print(json.dumps(stats, indent=2))        
        episode_frames = group_frames_by_episodes(trajectories)
        return render_episode_frames(episode_frames, out_dir="./tmp", grid=(2, 2), text_color=(0, 0, 0)) # TODO: review if eval collector should be deterministic or not
