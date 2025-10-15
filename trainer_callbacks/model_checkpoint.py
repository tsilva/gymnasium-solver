"""Checkpoint management utilities for model saving and resuming training."""

import tempfile
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from utils.io import write_json
from utils.metrics_serialization import prepare_metrics_for_json
from utils.run import Run
from utils.scalars import only_scalar_values


class ModelCheckpointCallback(pl.Callback):
    """Custom checkpoint callback that handles all model checkpointing logic including resume."""

    def __init__(
        self,
        run: Run,
        metric: str,
        mode: str = "max",
    ):
        """Initialize checkpointing for the monitored metric."""

        # Store attributes
        self.run = run
        self.metric = metric
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.first_eval = True  # Track if this is the first evaluation

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._if_best_epoch_save_checkpoint(trainer, pl_module)

    def _if_best_epoch_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # If the monitored metric wasn't logged this epoch (e.g., warmup), skip
        if self.metric not in trainer.logged_metrics: return

        # Determine if this is the best epoch so far
        metrics = only_scalar_values(trainer.logged_metrics)
        metric_value = metrics[self.metric]
        is_best = (metric_value > self.best_value) if (self.mode == "max") else (metric_value < self.best_value)

        # Save checkpoint if:
        # - This is the first eval (no basis for comparison), OR
        # - This is the best so far, OR
        # - Training is stopping (early stop, max steps, etc.)
        should_save = self.first_eval or is_best or trainer.should_stop

        # If none of the conditions are met, return (do nothing)
        if not should_save: return

        # Save checkpoint with video recorded into checkpoint dir
        epoch = pl_module.current_epoch

        # Determine if this checkpoint should be marked as best
        # (either genuinely best, or first eval which serves as initial best)
        mark_as_best = is_best or self.first_eval

        # Update best value if this is better (or if first eval)
        if mark_as_best:
            self.best_value = metric_value

        # Mark that we've completed first eval
        if self.first_eval:
            self.first_eval = False

        self._save_checkpoint(pl_module, epoch, metrics, is_best=mark_as_best)

    def _save_checkpoint(
        self,
        agent: pl.LightningModule,
        epoch: int,
        metrics: dict[str, Any],
        is_best: bool,
    ) -> tuple[Path, float]:
        """Save a checkpoint and return the path plus wall-clock save duration."""

        # Save checkpoint to temp dir and then let run pick up data from there
        # (this allows run to be agnostic to the contents of the checkpoint data)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            # Delegate checkpoint saving to agent (model, optimizer, RNG states, etc.)
            agent.save_checkpoint(tmp_dir)

            # Serialize metrics with configured precision and key priority
            json_metrics = prepare_metrics_for_json(metrics)
            write_json(tmp_dir / "metrics.json", json_metrics)

            # Recording best-checkpoint video is currently disabled.
            # Previous attempt ran an extra evaluation here, which stalled training throughput.
            # TODO: restore recording
            # if is_best:
            #     video_path = tmp_dir / "best.mp4"
            #     val_env = agent.get_env("val")
            #     with val_env.recorder(str(video_path), record_video=True):
            #         val_collector = agent.get_rollout_collector("val")
            #         val_collector.evaluate_episodes(
            #             n_episodes=1,
            #             deterministic=agent.config.eval_deterministic,
            #         )


            self.run.save_checkpoint(epoch, tmp_dir, is_best=is_best)
