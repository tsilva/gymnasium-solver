import torch
import numpy as np
from typing import Optional, Tuple, Iterator, Dict, Any
from torch.utils.data import Dataset as TorchDataset, DataLoader
from collections import deque
from contextlib import contextmanager
import itertools


@contextmanager
def inference_ctx(*modules):
    """Context manager for inference mode with model eval state management."""
    flat = [m for m in itertools.chain.from_iterable(
            (m if isinstance(m, (list, tuple)) else (m,)) for m in modules)
            if m is not None]
    
    was_training = [m.training for m in flat]
    try:
        for m in flat:
            m.eval()
        with torch.inference_mode():
            yield
    finally:
        for m, flag in zip(flat, was_training):
            if flag:
                m.train()


class RolloutBuffer:
    """Simple buffer that holds the most recent rollout data."""
    
    def __init__(self):
        self.data = None
        self.stats = None
    
    def update(self, trajectories: Tuple, stats: Dict[str, Any]):
        """Update buffer with new rollout data."""
        self.data = trajectories
        self.stats = stats
    
    def get_data(self) -> Tuple:
        """Get the current rollout data."""
        if self.data is None:
            raise ValueError("No rollout data available. Call collect() first.")
        return self.data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get the current rollout stats."""
        if self.stats is None:
            raise ValueError("No rollout stats available. Call collect() first.")
        return self.stats
    
    def create_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from the current rollout data."""
        if self.data is None:
            raise ValueError("No rollout data available. Call collect() first.")
        
        dataset = _RolloutDataset(self.data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class _RolloutDataset(TorchDataset):
    """Internal dataset class for rollout data."""
    
    def __init__(self, trajectories: Tuple):
        self.trajectories = trajectories
    
    def __len__(self):
        return len(self.trajectories[0])
    
    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.trajectories)


class RolloutCollector:
    """
    Cleaner rollout collector that separates concerns:
    - Collection logic
    - Data storage
    - DataLoader creation (when needed)
    """
    
    def __init__(
        self,
        env,
        policy_model: torch.nn.Module,
        value_model: Optional[torch.nn.Module] = None,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
        deterministic: bool = False,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.deterministic = deterministic
        self.gamma = gamma
        self.lam = lam
        
        # Use a simple buffer instead of maintaining generator state
        self.buffer = RolloutBuffer()
        
        # Stats tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def collect(
        self,
        n_steps: Optional[int] = None,
        n_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """
        Collect a fresh rollout and update the buffer.
        Returns trajectories and stats.
        """
        # Use provided values or fall back to defaults
        n_steps = n_steps if n_steps is not None else self.n_steps
        n_episodes = n_episodes if n_episodes is not None else self.n_episodes
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Collect rollout using the existing function
        from . import rollouts  # Import the original collect_rollouts function
        generator = rollouts.collect_rollouts(
            self.env,
            self.policy_model,
            value_model=self.value_model,
            n_steps=n_steps,
            n_episodes=n_episodes,
            deterministic=deterministic,
            gamma=self.gamma,
            lam=self.lam,
        )
        
        trajectories, stats = next(generator)
        
        # Update buffer
        self.buffer.update(trajectories, stats)
        
        # Update stats tracking
        if 'mean_ep_reward' in stats:
            # This is a simplification - you might want to track individual episodes
            pass
        
        return trajectories, stats
    
    def get_current_data(self) -> Tuple:
        """Get the most recently collected rollout data."""
        return self.buffer.get_data()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get the most recently collected rollout stats."""
        return self.buffer.get_stats()
    
    def create_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from the current rollout data."""
        return self.buffer.create_dataloader(batch_size, shuffle)


# Updated base learner that doesn't force the dataset/dataloader pattern
class ImprovedLearner:
    """
    Alternative learner base class that works more naturally with RL.
    Can still inherit from pl.LightningModule if needed.
    """
    
    def __init__(
        self,
        config,
        train_collector: RolloutCollector,
        policy_model: torch.nn.Module,
        value_model: Optional[torch.nn.Module] = None,
        eval_collector: Optional[RolloutCollector] = None,
    ):
        self.config = config
        self.train_collector = train_collector
        self.eval_collector = eval_collector
        self.policy_model = policy_model
        self.value_model = value_model
    
    def training_step(self) -> Dict[str, Any]:
        """
        One training step:
        1. Collect rollout if needed
        2. Create dataloader from rollout
        3. Train on batches
        4. Return metrics
        """
        # Collect fresh rollout
        _, stats = self.train_collector.collect()
        
        # Create dataloader for this rollout
        dataloader = self.train_collector.create_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Train on batches
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            loss_dict = self.compute_loss(batch)
            self.optimize_models(loss_dict)
            total_loss += loss_dict.get('total_loss', 0.0)
            num_batches += 1
        
        # Return combined metrics
        metrics = {
            'avg_batch_loss': total_loss / num_batches if num_batches > 0 else 0.0,
            **stats  # Include rollout stats
        }
        
        return metrics
    
    def compute_loss(self, batch) -> Dict[str, torch.Tensor]:
        """Override in subclass."""
        raise NotImplementedError
    
    def optimize_models(self, loss_dict: Dict[str, torch.Tensor]):
        """Override in subclass."""
        raise NotImplementedError


# Alternative: Lightning-compatible version that's cleaner
class LightningRLLearner:
    """
    PyTorch Lightning compatible version that doesn't abuse the dataloader pattern.
    """
    
    def __init__(self, config, train_collector: RolloutCollector, **kwargs):
        # ... same init as before
        self.train_collector = train_collector
        self.config = config
        self._current_dataloader = None
    
    def train_dataloader(self):
        """
        Lightning requires this, but we'll make it cleaner.
        We collect rollout here and create a single-use dataloader.
        """
        if self._current_dataloader is None:
            # Collect rollout only when Lightning asks for dataloader
            _, stats = self.train_collector.collect()
            self._current_dataloader = self.train_collector.create_dataloader(
                batch_size=self.config.batch_size
            )
            # Log rollout stats
            self.log_dict(stats, prefix="rollout")
        
        return self._current_dataloader
    
    def on_train_epoch_start(self):
        """Reset dataloader for next epoch."""
        self._current_dataloader = None
    
    def training_step(self, batch, batch_idx):
        """Standard Lightning training step."""
        loss_dict = self.compute_loss(batch)
        self.optimize_models(loss_dict)
        
        # Log training metrics
        self.log_dict({k: v for k, v in loss_dict.items() if isinstance(v, torch.Tensor)})
        
        return loss_dict.get('total_loss', list(loss_dict.values())[0])
