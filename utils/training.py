"""Training setup utilities."""

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tsilva_notebook_utils.lightning import WandbCleanup
from learners.ppo import PPOLearner
from learners.reinforce import REINFORCELearner
from utils.rollouts import AsyncRolloutCollector, SyncRolloutCollector
from .models import PolicyNet, ValueNet


def create_agent(config, build_env_fn, obs_dim, act_dim, algorithm=None):
    """Create RL agent based on algorithm configuration."""
    
    # Create rollout collector env
    rollout_env = build_env_fn(config.seed + 1000, n_envs=config.n_envs)
    rollout_collector_cls = AsyncRolloutCollector if config.async_rollouts else SyncRolloutCollector
    
    algo_id = algorithm or config.algorithm # TODO: move algo out of config
    algo_id = algo_id.lower()
    policy_model = PolicyNet(obs_dim, act_dim, config.hidden_dims)
    value_model = ValueNet(obs_dim, config.hidden_dims) if algo_id == "ppo" else None
    rollout_collector = rollout_collector_cls(config, rollout_env, policy_model, value_model=value_model)
    agent = PPOLearner(config, build_env_fn, rollout_collector, policy_model, value_model) if algo_id == "ppo" else REINFORCELearner(config, build_env_fn, rollout_collector, policy_model)
    
    return agent


def create_trainer(config, project_name=None, run_name=None):
    """Create PyTorch Lightning trainer with W&B logging."""
    
    project_name = project_name or config.env_id
    run_name = run_name or f"{getattr(config, 'algorithm')}-{wandb.util.generate_id()[:5]}"
    
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        log_model=True
    )
    
    # Print W&B run URL explicitly
    print(f"🔗 W&B Run: {wandb_logger.experiment.url}")
    
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=config.max_epochs,
        enable_progress_bar=True,
        enable_checkpointing=False,  # Disable checkpointing for speed
        accelerator="auto",
        callbacks=[WandbCleanup()]
    )
    
    return trainer

