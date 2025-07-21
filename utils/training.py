"""Training setup utilities."""

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tsilva_notebook_utils.lightning import WandbCleanup
from utils.rollouts import SyncRolloutCollector

    

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

