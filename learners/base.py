import time
import numpy as np
import pytorch_lightning as pl
from utils.rollouts import (group_trajectories_by_episode)
#from tsilva_notebook_utils.gymnasium import MetricTracker

# ---------------------------------------------------------------------
class Learner(pl.LightningModule):
    """Base agent class with common RL functionality"""
    
    def __init__(self, config, train_rollout_collector, policy_model, value_model=None, eval_rollout_collector=None):
        super().__init__()
        
        # Store core attributes
        self.config = config
        
        # Common RL components
        self.train_rollout_collector = train_rollout_collector
        self.eval_rollout_collector = eval_rollout_collector
        #self.metrics = MetricTracker(self)

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

    def setup(self, stage: str):
        if stage == "fit": self._setup_stage_fit()
    
    def _setup_stage_fit(self):
        self.train_rollout_collector.collect_rollouts()

    def train_dataloader(self):
        return self.train_rollout_collector.create_dataloader(self.config.batch_size)

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
        if (self.current_epoch + 1) % self.config.rollout_interval == 0:
            self.train_rollout_collector.collect_rollouts()
            #self._collect_and_update_rollout()

    def on_train_epoch_end(self):
        # Log epoch metrics
        #epoch_metrics = self.metrics.compute_epoch_means()
        #if epoch_metrics:
        #    self.metrics.log_metrics(epoch_metrics, prefix="epoch")
        
        # Evaluation
        if (self.current_epoch + 1) % self.config.eval_interval == 0: self._check_eval_early_stop()

    # TODO: log training step duration
    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch

        # Track total steps consumed by trainer
        batch_size = states.size(0)
        self.total_steps += batch_size

        # Compute algorithm-specific losses and metrics
        loss_results = self.compute_loss(batch)

        # Track step metrics
        #self.metrics.add_step_metrics(loss_results)

        # Optimize models
        self.optimize_models(loss_results)

        # Log training metrics
        self._log_training_metrics(loss_results, advantages, values, returns)

        return sum(loss for key, loss in loss_results.items() if 'loss' in key)
    
    # TODO: make all training metrics be logged in end of epoch
    def _log_training_metrics(self, loss_results, advantages, values, returns):
        """Log common training metrics"""
       # mean_reward = np.mean(self.episode_reward_deque) if len(self.episode_reward_deque) > 0 else 0
        
        # Core metrics that most algorithms will have
        train_metrics = {
           # 'mean_reward': mean_reward,
            'total_steps': self.total_steps,
        }
        
        # Add algorithm-specific loss metrics
        for key, value in loss_results.items():
            if 'loss' in key or key in ['entropy', 'kl_divergence', 'explained_variance']:
                train_metrics[key] = value
        
        additional_metrics = {
            'advantage_mean': advantages.mean(),
            'advantage_std': advantages.std(),
            'value_mean': values.mean(),
            'returns_mean': returns.mean(),
        }
        
        # Add any additional algorithm-specific metrics
        for key, value in loss_results.items():
            if key not in train_metrics and key not in additional_metrics:
                additional_metrics[key] = value
        
        #self.metrics.log_metrics(train_metrics, prefix="train", prog_bar=True)
        #self.metrics.log_metrics(additional_metrics, prefix="train", prog_bar=False)

    # TODO: this measurement is incorrect because we get different number of episodes each time
    # TODO: drop most of this codebase, just want to pass rollout collector inside
    def _check_eval_early_stop(self):
        trajectories = None
        while trajectories is None: trajectories = self.eval_rollout_collector.collect_rollouts() # TODO: move this loop inside the collector
        episodes = group_trajectories_by_episode(trajectories, max_episodes=self.config.eval_episodes)
        episode_rewards = [sum(step[2] for step in episode) for episode in episodes]
        mean_episode_reward = np.mean(episode_rewards)
        
        # TODO: move metric tracking inside rollout collector
        # Log rollout metrics
        rollout_metrics = {
            'mean_reward': mean_episode_reward,
            'num_episodes': len(episodes),
            'num_steps': len(trajectories[0]),
            'avg_steps_per_episode': len(trajectories[0]) / (len(episodes) + 1e-3),
            #'time_elapsed': elapsed,
            #'steps_per_second': len(trajectories[0]) / (elapsed + 1e-3)
        }
        #self.metrics.log_metrics(rollout_metrics, prefix="eval")

        # TODO: can't log twice
        # self.metrics.log_single('eval/mean_reward', mean_episode_reward, prog_bar=True)

        if mean_episode_reward >= self.config.reward_threshold:
            print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {mean_episode_reward:.2f} >= threshold {self.config.reward_threshold}")
            self.trainer.should_stop = True

        return mean_episode_reward
        

