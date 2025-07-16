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
    
    # Determine algorithm - can be passed explicitly or derived from config
    if algorithm is None:
        # Try to get algorithm from config, default to PPO
        algo_id = getattr(config, 'algorithm', 'PPO').upper()
    else:
        algo_id = algorithm.upper()
    
    # Create models
    policy_model = PolicyNet(obs_dim, act_dim, config.hidden_dim)
    
    # Create rollout collector
    rollout_env = build_env_fn(config.seed + 1000, n_envs=config.n_envs)
    rollout_collector_cls = AsyncRolloutCollector if config.async_rollouts else SyncRolloutCollector
    
    if algo_id == "PPO":
        value_model = ValueNet(obs_dim, config.hidden_dim)
        rollout_collector = rollout_collector_cls(config, rollout_env, policy_model, value_model=value_model)
        agent = PPOLearner(config, build_env_fn, rollout_collector, policy_model, value_model)
    elif algo_id == "REINFORCE":
        rollout_collector = rollout_collector_cls(config, rollout_env, policy_model, value_model=None)
        agent = REINFORCELearner(config, build_env_fn, rollout_collector, policy_model)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_id}. Choose 'PPO' or 'REINFORCE'")
    
    return agent


def create_trainer(config, project_name=None, run_name=None):
    """Create PyTorch Lightning trainer with W&B logging."""
    
    project_name = project_name or config.env_id
    run_name = run_name or f"{getattr(config, 'algorithm', 'PPO')}-{wandb.util.generate_id()[:5]}"
    
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


def get_monitoring_info():
    """Get information about key metrics to monitor during training."""
    
    primary_metrics = [
        "eval/mean_reward",           # Main success indicator
        "train/mean_reward",          # Training progress
        "epoch/explained_var",        # Value function quality
        "epoch/entropy",              # Exploration level
        "epoch/clip_fraction"         # Policy update stability
    ]

    warning_conditions = {
        "epoch/clip_fraction > 0.5": "Reduce policy_lr or clip_epsilon",
        "epoch/approx_kl > 0.1": "Reduce policy_lr", 
        "epoch/explained_var < 0.3": "Increase value_lr or network size",
        "epoch/entropy < 0.01": "Increase entropy_coef",
        "rollout/queue_miss > rollout/queue_updated": "Check async collection"
    }
    
    return {
        "primary_metrics": primary_metrics,
        "warning_conditions": warning_conditions
    }
