"""Builder for trainer loggers."""

from dataclasses import asdict
from typing import TYPE_CHECKING

import wandb
from pytorch_lightning.loggers import WandbLogger

from loggers.metrics_csv_lightning_logger import MetricsCSVLightningLogger
from loggers.metrics_table_logger import MetricsTableLogger
from utils.formatting import sanitize_name
from utils.metrics_config import metrics_config

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


class TrainerLoggersBuilder:
    """Builds trainer loggers for an agent."""

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self.config = agent.config
        self.run = agent.run
        self.metrics_monitor = agent.metrics_monitor

    def build(self) -> list:
        """Build all trainer loggers."""
        loggers = []

        # Prepare a wandb logger (only if enabled)
        if getattr(self.config, "enable_wandb", True):
            wandb_logger = self._build_wandb_logger()
            loggers.append(wandb_logger)

        # Prepare a CSV Lightning logger writing to runs/<id>/metrics.csv
        csv_logger = self._build_csv_logger()
        loggers.append(csv_logger)

        # Prepare a terminal print logger that formats metrics from the unified logging stream (skip if quiet)
        if not getattr(self.config, "quiet", False):
            print_logger = self._build_print_logger()
            loggers.append(print_logger)

        return loggers

    def _build_wandb_logger(self) -> WandbLogger:
        """Build W&B logger."""
        # Create the wandb logger, attach to the existing run if present
        project_name = (
            self.config.project_id if self.config.project_id else sanitize_name(self.config.env_id)
        )
        experiment_name = f"{self.config.algo_id}-{self.config.seed}"
        wandb_logger = (
            WandbLogger(
                project=project_name,
                name=experiment_name,
                log_model=True,
                config=asdict(self.config),
            )
            if wandb.run is None
            else WandbLogger(log_model=True)
        )

        # Define the common step metric
        wandb_run = wandb_logger.experiment
        wandb_run.define_metric("*", step_metric=metrics_config.total_timesteps_key())

        # Change the run name to {algo_id}-{run_id}
        wandb_run.name = f"{self.config.algo_id}-{wandb_run.id}"

        # Log model gradients to wandb
        wandb_logger.watch(self.agent.policy_model, log="gradients", log_freq=100)

        return wandb_logger

    def _build_csv_logger(self) -> MetricsCSVLightningLogger:
        """Build CSV logger."""
        csv_path = self.run.ensure_metrics_path()
        csv_logger = MetricsCSVLightningLogger(csv_path=str(csv_path))
        return csv_logger

    def _build_print_logger(self) -> MetricsTableLogger:
        """Build print logger."""
        print_logger = MetricsTableLogger(metrics_monitor=self.metrics_monitor, run=self.run)
        return print_logger
