import time
import multiprocessing
from collections import deque
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.rollouts import (
    collect_rollouts, group_trajectories_by_episode,
    RolloutDataset, AsyncRolloutCollector, SyncRolloutCollector
)
from tsilva_notebook_utils.gymnasium import MetricTracker

# ---------------------------------------------------------------------
class Learner(pl.LightningModule):
    """Base agent class with common RL functionality"""
    
    def __init__(self, config, build_env_fn, rollout_collector, policy_model, value_model=None):
        super().__init__()
        
        # Store core attributes
        self.config = config
        
        # Common RL components
        self.build_env_fn = build_env_fn
        self.rollout_collector = rollout_collector
        self.metrics = MetricTracker(self)
        self.rollout_ds = RolloutDataset()
        self.episode_reward_deque = deque(maxlen=config.mean_reward_window)
        
        self.policy_model = policy_model
        self.value_model = value_model
        
        # Training state
        self.automatic_optimization = False
        self.training_start_time = None
        self.total_steps = 0  # Track total training steps consumed
        
    def compute_loss(self, batch):
        """Override in subclass to compute algorithm-specific loss"""
        raise NotImplementedError("Subclass must implement compute_loss()")
        
    def optimize_models(self, loss_results):
        """Override in subclass to implement algorithm-specific optimization"""
        raise NotImplementedError("Subclass must implement optimize_models()")

    def setup(self, stage: str):
        if stage == "fit":
            self.rollout_collector.start()
            
            print("Waiting for initial rollout...")
            while True:
                trajectories = self.rollout_collector.get_rollout(timeout=2.0)
                if trajectories is None: print("Still waiting for rollout..."); continue
                self._update_rollout_data(trajectories)
                break
                
    def train_dataloader(self):
        return DataLoader(
            self.rollout_ds,
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
        self.rollout_collector.stop()
        if self.training_start_time:
            total_time = time.time() - self.training_start_time
            print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    def on_train_epoch_start(self):
        self.metrics.reset()
        self.rollout_collector.update_models(
            self.policy_model.state_dict(), self.value_model.state_dict() if self.value_model else None
        )
        
        # Collect new rollout if needed
        if (self.current_epoch + 1) % self.config.rollout_interval == 0:
            self._collect_and_update_rollout()

    def on_train_epoch_end(self):
        # Log epoch metrics
        epoch_metrics = self.metrics.compute_epoch_means()
        if epoch_metrics:
            self.metrics.log_metrics(epoch_metrics, prefix="epoch")
        
        # Evaluation
        if (self.current_epoch + 1) % self.config.eval_interval == 0:
            self._evaluate_and_check_stopping()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, old_logps, values, advantages, returns, frames = batch

        # Track total steps consumed by trainer
        batch_size = states.size(0)
        self.total_steps += batch_size

        # Compute algorithm-specific losses and metrics
        loss_results = self.compute_loss(batch)

        # Track step metrics
        self.metrics.add_step_metrics(loss_results)

        # Optimize models
        self.optimize_models(loss_results)

        # Log training metrics
        self._log_training_metrics(loss_results, advantages, values, returns)

        return sum(loss for key, loss in loss_results.items() if 'loss' in key)

    def _collect_and_update_rollout(self):
        """Collect and update rollout data"""
        timeout = 2.0 if self.config.async_rollouts else 1.0
        trajectories = self.rollout_collector.get_rollout(timeout=timeout)
        
        if trajectories is not None:
            self._update_rollout_data(trajectories)
            self.metrics.log_single('rollout/queue_updated', 1.0)
        else:
            self.metrics.log_single('rollout/queue_miss', 1.0)
    
    def _update_rollout_data(self, trajectories):
        """Update rollout dataset and episode rewards"""
        self.rollout_ds.update(*trajectories)
        episodes = group_trajectories_by_episode(trajectories)
        episode_rewards = [sum(step[2] for step in episode) for episode in episodes]
        for r in episode_rewards:
            self.episode_reward_deque.append(float(r))

    def _log_training_metrics(self, loss_results, advantages, values, returns):
        """Log common training metrics"""
        mean_reward = np.mean(self.episode_reward_deque) if len(self.episode_reward_deque) > 0 else 0
        
        # Core metrics that most algorithms will have
        train_metrics = {
            'mean_reward': mean_reward,
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
        
        self.metrics.log_metrics(train_metrics, prefix="train", prog_bar=True)
        self.metrics.log_metrics(additional_metrics, prefix="train", prog_bar=False)

    def _evaluate_and_check_stopping(self):
        """Evaluate model and check for early stopping"""
        eval_seed = np.random.randint(0, 1_000_000)
        
        # CRITICAL FIX: Force single environment for evaluation
        # 
        # PROBLEM: When n_envs="auto" in config, build_env_fn creates vectorized environments 
        # with multiple parallel environments (e.g., 8 envs on an 8-core CPU). During evaluation,
        # collect_rollouts(n_episodes=5) would collect 5 episodes TOTAL across ALL environments,
        # not 5 episodes per environment. This caused several issues:
        # 
        # 1. INCOMPLETE EPISODES: With 8 parallel envs and only 5 total episodes requested,
        #    some environments would have incomplete episodes that still contributed to rewards
        # 2. INFLATED REWARDS: Reward calculation summed across all timesteps rather than 
        #    properly grouping by complete episodes, leading to values like 807.80 instead 
        #    of the maximum 500 for CartPole-v1
        # 3. INCORRECT EARLY STOPPING: Artificially high evaluation rewards triggered
        #    early stopping when the agent wasn't actually performing well
        #
        # ROOT CAUSE: The vectorized environment's episode handling doesn't align with
        # how collect_rollouts counts and groups episodes for reward calculation.
        #
        # FIX: Override n_envs=1 for evaluation to ensure we get exactly the requested
        # number of complete episodes with proper reward calculation per episode.
        eval_env = self.build_env_fn(eval_seed, n_envs=1)
        
        self.policy_model.eval()
        if self.value_model: self.value_model.eval()
        try:
            eval_mean_reward = self._run_evaluation(eval_env, self.policy_model, self.value_model)
            self.metrics.log_single('eval/mean_reward', eval_mean_reward, prog_bar=True)
            
            if eval_mean_reward >= self.config.reward_threshold:
                print(f"Early stopping at epoch {self.current_epoch} with eval mean reward {eval_mean_reward:.2f} >= threshold {self.config.reward_threshold}")
                self.trainer.should_stop = True
                
        finally:
            self.policy_model.train()
            if self.value_model: self.value_model.train()
            eval_env.close()

    def _run_evaluation(self, env, policy_model, value_model):
        """Run evaluation and log rollout metrics"""
        start = time.time()
        trajectories, _ = collect_rollouts(
            env, policy_model, value_model,
            n_episodes=self.config.eval_episodes, deterministic=False
        )
        elapsed = time.time() - start

        episodes = group_trajectories_by_episode(trajectories)
        episode_rewards = [sum(step[2] for step in episode) for episode in episodes]
        mean_episode_reward = np.mean(episode_rewards)
        
        # Log rollout metrics
        rollout_metrics = {
            'mean_reward': mean_episode_reward,
            'num_episodes': len(episodes),
            'num_steps': len(trajectories[0]),
            'avg_steps_per_episode': len(trajectories[0]) / (len(episodes) + 1e-3),
            'time_elapsed': elapsed,
            'steps_per_second': len(trajectories[0]) / (elapsed + 1e-3)
        }
        
        self.metrics.log_metrics(rollout_metrics, prefix="rollout")
        return mean_episode_reward
