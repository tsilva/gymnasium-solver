"""Callback to optionally zip and upload run folder to W&B after training completes."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytorch_lightning as pl
from tqdm import tqdm

import wandb


class UploadRunCallback(pl.Callback):
    """Upload zipped run folder to W&B after training completes.

    Prompts the user after training ends, and if confirmed:
    1. Creates a zip archive of the run folder with progress bar
    2. Uploads the archive to W&B as an artifact with progress tracking
    3. Cleans up the temporary zip file
    """

    def __init__(self, run_dir: Path):
        super().__init__()
        self.run_dir = Path(run_dir)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Hook called when training completes."""
        # Only proceed if W&B is active
        run = getattr(wandb, "run", None)
        if not run:
            return

        # Prompt user if they want to upload
        from utils.user import prompt_confirm
        should_upload = prompt_confirm("Upload run folder to W&B?", default=False)
        if not should_upload:
            print("Skipping run folder upload.")
            return

        # Create temporary zip file
        run_id = self.run_dir.name
        temp_dir = Path(tempfile.gettempdir())
        zip_path = temp_dir / f"{run_id}.zip"

        # Count total files for progress tracking
        print("Counting files to archive...")
        all_files = list(self.run_dir.rglob("*"))
        file_paths = [f for f in all_files if f.is_file()]
        total_files = len(file_paths)

        # Create zip archive with progress bar
        print(f"Creating archive of {total_files} files...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            with tqdm(total=total_files, desc="Zipping", unit="file") as pbar:
                for file_path in file_paths:
                    arcname = file_path.relative_to(self.run_dir.parent)
                    zipf.write(file_path, arcname)
                    pbar.update(1)

        # Get file size for upload progress
        zip_size = zip_path.stat().st_size
        zip_size_mb = zip_size / (1024 * 1024)
        print(f"Archive created: {zip_path.name} ({zip_size_mb:.1f} MB)")

        # Upload to W&B as artifact
        print("Uploading to Weights & Biases...")
        artifact = wandb.Artifact(
            name=f"run-{run_id}",
            type="run-archive",
            description=f"Complete run folder archive for {run_id}",
        )

        # Add the zip file to the artifact
        # W&B will show upload progress automatically
        artifact.add_file(str(zip_path), name=f"{run_id}.zip")

        # Log the artifact (this triggers the upload with progress)
        run.log_artifact(artifact)

        print(f"✓ Run folder uploaded to W&B as artifact 'run-{run_id}'")

        # Clean up temporary zip file
        zip_path.unlink()
