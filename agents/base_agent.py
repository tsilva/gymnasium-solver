from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import threading

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn

from agents import base_agent_runtime
from agents.base_agent_async_eval import cleanup_async_eval, launch_async_eval
from agents.base_agent_bootstrap import build_stage_env, build_stage_rollout_collector
from agents.base_agent_checkpointing import (
    load_agent_checkpoint,
    save_agent_checkpoint,
)
from agents.base_agent_optimization import backpropagate_and_step
from agents.hyperparameter_mixin import HyperparameterMixin
from utils.config import Config
from utils.decorators import must_implement
from utils.metric_bundles import CoreMetricAlerts
from utils.metrics_monitor import MetricsMonitor
from utils.metrics_recorder import MetricsRecorder
from utils.rollout_buffer import RolloutTrajectory
from utils.rollout_collector import RolloutCollector
from utils.run import Run
from utils.timings_tracker import TimingsTracker

if TYPE_CHECKING:
    from loggers.metrics_table_logger import MetricsTableLogger

STAGES = ["train", "val", "test"]

class BaseAgent(HyperparameterMixin, pl.LightningModule):
    
    run: Run
    config: Config
    policy_model: nn.Module
    _envs: Dict[str, Any]
    _rollout_collectors: Dict[str, RolloutCollector]
    _trajectories: List[RolloutTrajectory]
    _early_stop_epoch: bool
    _fit_elapsed_seconds: float
    _final_stop_reason: str
    _train_dataloader: DataLoader
    _async_eval_thread: Optional[threading.Thread]
    _async_eval_metrics: Dict[str, Any]
    _async_eval_lock: threading.Lock

    def __init__(self, config):
        super().__init__()

        # Define which hyperparameters are saved along 
        # with module checkpoints (default to all kwargs)
        self.save_hyperparameters()

        # Disable automatic optimization to allow manual optimization, namely for tuning 
        # multiple models with different optimizers (eg: independent policy and value models)
        self.automatic_optimization = False

        # Initialize instance attributes
        self.config = config
        self.run = None
        self.policy_model = None
        self._envs = {}
        self._rollout_collectors = {}
        self._trajectories = []
        self._early_stop_epoch = False
        self._fit_elapsed_seconds = 0.0
        self._final_stop_reason = ""
        self._early_stop_reason = ""
        self._print_metrics_logger: Optional["MetricsTableLogger"] = None
        self._async_eval_thread = None
        self._async_eval_metrics = {}
        self._async_eval_lock = threading.Lock()
        self._async_eval_shutdown = threading.Event()
        self._async_eval_pending_epoch = None  # Tracks if we need to eval a newer model
        self._async_eval_running_epoch = None  # Tracks which epoch is currently being evaluated

        # Initialize schedulable hyperparameters (mutable during training)
        self.policy_lr = config.policy_lr
        self.clip_range = config.clip_range
        self.vf_coef = config.vf_coef
        self.ent_coef = config.ent_coef
        self.n_epochs = config.n_epochs
        
        # Initialize timing tracker for training 
        # loop performance measurements
        self.timings = TimingsTracker()

        # Metrics recorder aggregates per-epoch metrics (train/eval) and maintains
        # a step-aware numeric history for terminal summaries.
        self.metrics_recorder = MetricsRecorder()

        # Create metrics monitor registry (eg: used for metric alerts)
        self.metrics_monitor = MetricsMonitor(self.metrics_recorder)

        # Register bundle of metric alerts that apply to all algorithms
        core_metric_alerts = CoreMetricAlerts(self)
        self.metrics_monitor.register_bundle(core_metric_alerts)

        # Build the environments
        for stage in STAGES: self.build_env(stage)

        # Build the models (requires environments for shape inference)
        self.build_models()

        # Build the rollout collectors (requires models and environments)
        for stage in STAGES: self.build_rollout_collector(stage)

    def build_models(self):
        from utils.policy_factory import build_policy_from_env_and_config
        train_env = self.get_env("train")
        self.policy_model = build_policy_from_env_and_config(train_env, self.config)

    @must_implement
    def losses_for_batch(self, batch, batch_idx):
        # Subclasses must implement this to compute the losses for each training steps' batch
        pass

    def _normalize_advantages(self, advantages):
        """Normalize advantages if configured, returning normalized tensor and metrics."""
        from utils.torch import normalize_batch_with_metrics
        return normalize_batch_with_metrics(
            advantages, self.config.normalize_advantages, "roll/adv"
        )

    def _build_common_metrics(self, loss, policy_loss, entropy_loss, entropy):
        """Build common metrics shared across algorithms."""
        return {
            'opt/loss/total': loss.detach(),
            'opt/loss/policy': policy_loss.detach(),
            'opt/loss/entropy': entropy_loss.detach(),
            'opt/policy/entropy': entropy.detach(),
        }

    def build_env(self, stage: str, **kwargs):
        build_stage_env(self, stage, **kwargs)
            
    def get_env(self, stage: str):
        return self._envs[stage]

    def build_rollout_collector(self, stage: str):
        build_stage_rollout_collector(self, stage)

    def attach_print_metrics_logger(self, logger: "MetricsTableLogger") -> None:
        self._print_metrics_logger = logger

    def _set_stage_display(self, stage: str) -> None:
        if self._print_metrics_logger is None:
            return
        self._print_metrics_logger.set_stage(stage)

    def get_rollout_collector(self, stage: str):
        return self._rollout_collectors[stage]

    def on_fit_start(self):
        base_agent_runtime.on_fit_start(self)

    def train_dataloader(self):
        return base_agent_runtime.train_dataloader(self)

    def on_train_epoch_start(self):
        base_agent_runtime.on_train_epoch_start(self)

    def training_step(self, batch, batch_idx):
        return base_agent_runtime.training_step(self, batch, batch_idx)

    def on_train_epoch_end(self):
        # Scheduling moved to HyperparameterScheduler callback
        pass
        
    def val_dataloader(self):
        return base_agent_runtime.val_dataloader(self)

    def on_validation_epoch_start(self):
        base_agent_runtime.on_validation_epoch_start(self)

    def _launch_async_eval(self, eval_epoch: Optional[int] = None):
        launch_async_eval(self, eval_epoch=eval_epoch)

    # TODO: if running in bg, consider using simple rollout collector that sends metrics over, if eval mean_reward_treshold is reached, training is stopped
    # TODO: currently recording more than the requested episodes (rollout not trimmed)
    # TODO: there are train/fps drops caused by running the collector N times (its not only the video recording); cause currently unknown
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return base_agent_runtime.validation_step(
            self,
            batch,
            batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_epoch_end(self):
        base_agent_runtime.on_validation_epoch_end(self)

    def _cleanup_async_eval(self):
        cleanup_async_eval(self)

    def on_exception(self, trainer, pl_module, exception):
        """Handle cleanup when training fails with an exception."""
        self._cleanup_async_eval()

    def on_fit_end(self):
        base_agent_runtime.on_fit_end(self)
    
    def learn(self):
        base_agent_runtime.learn(self)

    def _learn(self):
        base_agent_runtime._learn(self)


    def _backpropagate_and_step(self, losses):
        backpropagate_and_step(self, losses)

    def calc_training_progress(self):
        """Calculate training progress as a fraction of max_env_steps (0.0 to 1.0)."""
        max_env_steps = self.config.max_env_steps
        if max_env_steps is None: return 0.0
        train_collector = self.get_rollout_collector("train")
        # max_env_steps is env_steps, so use total_steps for progress calculation
        total_env_steps = train_collector.total_steps
        training_progress = max(0.0, min(total_env_steps / max_env_steps, 1.0))
        return training_progress

    def configure_optimizers(self):
        from utils.optimizer_factory import build_optimizer
        return build_optimizer(
            params=self.policy_model.parameters(),
            optimizer=self.config.optimizer,
            lr=self.policy_lr, # TODO: is this taking annealing into account?
        )

    # -------------------------
    # Public API for callbacks
    # -------------------------

    def set_early_stop_reason(self, reason: str) -> None:
        """Set the early stopping reason. Called by EarlyStoppingCallback."""
        self._early_stop_reason = reason

    def get_async_eval_metric(self, metric_key: str) -> Optional[float]:
        """Get a metric from async eval results. Returns None if not available."""
        with self._async_eval_lock:
            return self._async_eval_metrics.get(metric_key)

    # -------------------------
    # Checkpoint save/load
    # -------------------------

    def save_checkpoint(self, checkpoint_dir: Path) -> None:
        save_agent_checkpoint(self, checkpoint_dir)

    def load_checkpoint(self, checkpoint_dir: Path, resume_training: bool = True, strict: bool = True, load_optimizer_only: bool = False) -> None:
        load_agent_checkpoint(
            self,
            checkpoint_dir,
            resume_training=resume_training,
            strict=strict,
            load_optimizer_only=load_optimizer_only,
        )
