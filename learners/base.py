import time
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

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
class Learner(pl.LightningModule):
    
    def __init__(self, config, train_rollout_collector, policy_model, value_model=None, eval_rollout_collector=None):
        super().__init__()
        
        # Store core attributes
        self.config = config
        
        # Common RL components
        self.train_rollout_collector = train_rollout_collector
        self.eval_rollout_collector = eval_rollout_collector

        self.policy_model = policy_model
        self.value_model = value_model
        
        self.dataset = RolloutDataset()

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
        self.dataset.update(*trajectories)
        self._train_rollout_stats = stats # TODO: do I need to do this

        # TODO: memory leaks? what if I initialize the dataloader with correct rollout size?
        # TODO: does this approach work where we swap the data underneath the dataloader between epochs?
        return DataLoader(
            self.dataset,
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

    # TODO: log train epoch duration
    def on_train_epoch_start(self):
        if self._train_rollout_stats:
            self.log_metrics(self._train_rollout_stats, prog_bar=["mean_ep_reward"], prefix="train")
            self._train_rollout_stats = None  # Reset stats after logging

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.config.eval_interval == 0: 
            _, stats = self.eval_rollout_collector.collect() # TODO: is this collecting expected number of episodes? assert mean reward is not greater than allowed by env
            self.log_metrics(stats, prog_bar=["mean_ep_reward"], prefix="eval")
            
            mean_ep_reward = stats['mean_ep_reward']
            if mean_ep_reward >= self.config.reward_threshold: # TODO; change to eval_reward_threshold
                print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {mean_ep_reward:.2f} >= threshold {self.config.reward_threshold}")
                self.trainer.should_stop = True

    def training_step(self, batch, batch_idx):
        batch_size = batch[0].size(0)
        self.total_steps += batch_size
        loss_results = self.compute_loss(batch)
        self.optimize_models(loss_results)

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
