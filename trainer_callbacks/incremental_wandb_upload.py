"""Callback to incrementally upload checkpoints to W&B during training."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl

import wandb


class IncrementalWandbUploadCallback(pl.Callback):
    """Upload checkpoints to W&B incrementally during training.

    This callback monitors the checkpoints directory and uploads new checkpoints
    to W&B as they're created. Uploads happen in a background thread to avoid
    blocking training. The W&B artifact mirrors the local run structure, enabling
    run_play.py to download runs that aren't available locally.
    """

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.uploaded_checkpoints = set()  # Track which checkpoint dirs we've uploaded
        self.upload_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.upload_queue = []  # Queue of checkpoint dirs to upload
        self.queue_lock = threading.Lock()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Hook called after validation completes - check for new checkpoints to upload."""
        # Only upload if W&B is active
        run = getattr(wandb, "run", None)
        if not run:
            return

        # Find new checkpoints that haven't been uploaded yet
        if not self.checkpoints_dir.exists():
            return

        for checkpoint_dir in self.checkpoints_dir.iterdir():
            # Skip symlinks (@best, @last) and non-directories
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                continue

            # Skip if already uploaded or queued
            if checkpoint_dir.name in self.uploaded_checkpoints:
                continue

            # Add to upload queue
            with self.queue_lock:
                if checkpoint_dir.name not in [p.name for p in self.upload_queue]:
                    self.upload_queue.append(checkpoint_dir)

        # Start upload thread if not already running and we have checkpoints to upload
        with self.queue_lock:
            has_work = len(self.upload_queue) > 0

        if has_work and (self.upload_thread is None or not self.upload_thread.is_alive()):
            self._start_upload_thread()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Wait for any pending uploads to complete when training ends."""
        # Signal shutdown
        self.shutdown_event.set()

        # Wait for upload thread to complete
        if self.upload_thread is not None and self.upload_thread.is_alive():
            print("Waiting for checkpoint uploads to complete...")
            self.upload_thread.join()
            print("✓ All checkpoint uploads completed")

    def _start_upload_thread(self):
        """Start background thread to process upload queue."""
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()

    def _upload_worker(self):
        """Worker function that processes the upload queue in the background."""
        while not self.shutdown_event.is_set():
            # Get next checkpoint to upload
            with self.queue_lock:
                if not self.upload_queue:
                    break
                checkpoint_dir = self.upload_queue.pop(0)

            # Upload the checkpoint
            try:
                self._upload_checkpoint(checkpoint_dir)
                self.uploaded_checkpoints.add(checkpoint_dir.name)
            except Exception as e:
                print(f"Warning: Failed to upload checkpoint {checkpoint_dir.name}: {e}")

    def _upload_checkpoint(self, checkpoint_dir: Path):
        """Upload a single checkpoint directory to W&B as part of the run artifact.

        Creates a complete run archive that mirrors the local structure, allowing
        run_play.py to download and use it seamlessly. Each upload creates a new
        version of the artifact with the latest checkpoint included.
        """
        run = wandb.run
        if not run:
            return

        # Create a temporary zip file containing the full run directory structure
        # (so that download_run_artifact can extract it directly)
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / f"{self.run_id}.zip"

            # Create zip archive with the full run directory
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add config.json (required for run loading)
                config_path = self.run_dir / "config.json"
                if config_path.exists():
                    arcname = Path(self.run_id) / "config.json"
                    zipf.write(config_path, arcname)

                # Add all checkpoint directories (including this new one)
                for ckpt_dir in self.checkpoints_dir.iterdir():
                    # Skip symlinks (@best, @last) - these will be recreated on download
                    if ckpt_dir.is_symlink():
                        continue

                    if ckpt_dir.is_dir():
                        # Add all files in this checkpoint directory
                        for file_path in ckpt_dir.rglob("*"):
                            if file_path.is_file():
                                arcname = Path(self.run_id) / "checkpoints" / ckpt_dir.name / file_path.relative_to(ckpt_dir)
                                zipf.write(file_path, arcname)

                # Optionally add metrics.csv if it exists (helpful for analysis)
                metrics_path = self.run_dir / "metrics.csv"
                if metrics_path.exists():
                    arcname = Path(self.run_id) / "metrics.csv"
                    zipf.write(metrics_path, arcname)

            # Upload to W&B artifact
            # Use the same artifact name as UploadRunCallback for consistency
            artifact = wandb.Artifact(
                name=f"run-{self.run_id}",
                type="run-archive",
                description=f"Training run archive for {self.run_id} (incremental update)",
            )

            artifact.add_file(str(zip_path), name=f"{self.run_id}.zip")

            # Log artifact (W&B will version this automatically)
            run.log_artifact(artifact)

            print(f"✓ Uploaded checkpoint {checkpoint_dir.name} to W&B")
