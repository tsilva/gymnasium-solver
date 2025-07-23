import time
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from utils.models import PolicyNet, ValueNet
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

        # TODO: should I set this before training starts? think deeply to where this should be set
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(config.seed)

        # Create environment builder
        from utils.environment import build_env
        build_env_fn = lambda seed: build_env(
            config.env_id,
            norm_obs=config.normalize,
            n_envs=n_envs,
            seed=seed
        )
        self.build_env_fn = build_env_fn
        
        # TODO: move this inside config?
        env = build_env_fn(config.seed)
        config.input_dim = env.observation_space.shape[0]
        config.output_dim = env.action_space.n

        policy_model = PolicyNet(config.input_dim, config.output_dim, config.hidden_dims)
        self.policy_model = policy_model

        # TODO: softcode this better
        value_model = ValueNet(config.input_dim, config.hidden_dims) if config.algo_id == "ppo" else None
        self.value_model = value_model

        # Common RL components
        self.train_rollout_collector = SyncRolloutCollector(
            build_env_fn(config.seed),
            policy_model,
            value_model=value_model,
            n_steps=config.train_rollout_steps
        )
        self.eval_rollout_collector = SyncRolloutCollector(
            # TODO: pass env factory and rebuild env on start/stop? this allows using same rollout collector for final evaluation
            build_env_fn(config.seed + 1000),
            policy_model,
            n_episodes=config.eval_episodes,
            deterministic=True
        )

        self.rollout_dataset = RolloutDataset()

        # Training state
        self.automatic_optimization = False
        self.training_start_time = None
        self.total_steps = 0  # Track total training steps consumed
        
    def forward(self, x):
        return self.policy_model(x)
    
    def compute_loss(self, batch):
        """Override in subclass to compute algorithm-specific loss"""
        raise NotImplementedError("Subclass must implement compute_loss()")
        
    def optimize_models(self, loss_results):
        """Override in subclass to implement algorithm-specific optimization"""
        raise NotImplementedError("Subclass must implement optimize_models()")

    def train_dataloader(self):
        trajectories, stats = self.train_rollout_collector.collect()
        self.rollout_dataset.update(*trajectories)
        self._train_rollout_stats = stats # TODO: do I need to do this

        # TODO: memory leaks? what if I initialize the dataloader with correct rollout size?
        # TODO: does this approach work where we swap the data underneath the dataloader between epochs?
        return DataLoader(
            self.rollout_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            # TODO: fix num_workers, training not converging when they are on
            # Pin memory is not supported on MPS
            #pin_memory=True if self.device.type != 'mps' else False,
            # TODO: Persistent workers + num_workers is fast but doesn't converge
            #persistent_workers=True if self.device.type != 'mps' else False,
            # Using multiple workers stalls the start of each epoch when persistent workers are disabled
            #num_workers=multiprocessing.cpu_count() // 2 if self.device.type != 'mps' else 0
        )

    def on_fit_start(self):
        self.training_start_time = time.time()
        self.total_steps = 0  # Reset step counter
        print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def on_fit_end(self):
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    def training_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        self.total_steps += batch_size

        loss_results = self.compute_loss(batch)

        self._train_rollout_stats = {**(self._train_rollout_stats or {}), **loss_results}

        self.optimize_models(loss_results)

    def on_train_epoch_end(self):
        if self._train_rollout_stats:
            self.log_metrics(self._train_rollout_stats, prog_bar=["mean_ep_reward"], prefix="train")
            self._train_rollout_stats = None  # Reset stats after logging

        if (self.current_epoch + 1) % self.config.eval_interval == 0: 
            _, stats = self.eval_rollout_collector.collect() # TODO: is this collecting expected number of episodes? assert mean reward is not greater than allowed by env
            self.log_metrics(stats, prog_bar=["mean_ep_reward"], prefix="eval")
            
            mean_ep_reward = stats['mean_ep_reward']
            if mean_ep_reward >= self.config.reward_threshold: # TODO; change to eval_reward_threshold
                print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {mean_ep_reward:.2f} >= threshold {self.config.reward_threshold}")
                self.trainer.should_stop = True

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
            reload_dataloaders_every_n_epochs=self.config.rollout_interval # TODO: train_rollout_interval
            #callbacks=[WandbCleanup()]
        )
        trainer.fit(self)


