import time
import pytorch_lightning as pl
import torch

def _device_of(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device

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

    #def setup(self, stage: str):
    #    if stage == "fit": self._setup_stage_fit()
    
    #def _setup_stage_fit(self):
        #self.train_rollout_collector.collect_rollouts()

    def train_dataloader(self):
        #device = _device_of(self.policy_model) # models are alredy in correct device ehre
        _, stats, dataloader = self.train_rollout_collector.create_dataloader(self.config.batch_size)
        #self.logger.experiment.log_dict(stats, on_step=False, on_epoch=True)#, prefix="train") # TODO: what are the on_step/on_epoch defaults?
        return dataloader

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
        #self.metrics.reset()
        
        # Collect new rollout if needed
        # TODO: hack, skipping rollout collection for first epoch because a rollout was collected when the dataloader was created (this seems like a hack)
        if self.current_epoch > 0 and (self.current_epoch + 1) % self.config.rollout_interval == 0:
            _, stats = self.train_rollout_collector.collect_rollouts() # TODO: prefix keys? BUG: this is discarding first rollout
            self.log_metrics(stats, prog_bar=["mean_ep_reward"], prefix="train")
            #self._collect_and_update_rollout()

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.config.eval_interval == 0: 
            _, stats = self.eval_rollout_collector.collect_rollouts() # TODO: is this collecting expected number of episodes? assert mean reward is not greater than allowed by env
            self.log_metrics(stats, prog_bar=["mean_ep_reward"], prefix="eval")
            mean_ep_reward = stats['mean_ep_reward']
            if mean_ep_reward >= self.config.reward_threshold:
                print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {mean_ep_reward:.2f} >= threshold {self.config.reward_threshold}")
                self.trainer.should_stop = True

    # TODO: log training step duration
    def training_step(self, batch, batch_idx): # TODO: should I call super?
        self.total_steps += batch[0].size(0)
        loss_results = self.compute_loss(batch)
        self.optimize_models(loss_results)

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    def log_metrics(self, metrics, *, on_epoch=None, on_step=None, prog_bar=None, prefix=None):
        for key, value in metrics.items():
            _on_epoch = True if on_epoch is True or key in (on_epoch or []) else None
            _on_step =  True if on_step is True or key in (on_step or []) else None
            _prog_bar = True if prog_bar is True or key in (prog_bar or []) else None
            _key = f"{prefix}/{key}" if prefix else key
            self.log(_key, value, on_epoch=_on_epoch, on_step=_on_step, prog_bar=_prog_bar)
