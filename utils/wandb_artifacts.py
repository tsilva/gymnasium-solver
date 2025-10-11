"""Utilities for downloading and managing W&B artifacts."""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Optional

import wandb
from dotenv import load_dotenv

# Load .env file if it exists (for WANDB_ENTITY, WANDB_PROJECT, etc.)
load_dotenv()


def _recreate_checkpoint_symlinks(run_dir: Path) -> None:
    """Recreate @best and @last symlinks in checkpoints directory.

    Symlinks are not preserved through zip/unzip, so we need to recreate them
    by analyzing checkpoint directories and their metrics.
    """
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return

    # Find all epoch checkpoint directories
    epoch_pattern = re.compile(r"^epoch=(\d+)$")
    checkpoints = []

    for path in checkpoints_dir.iterdir():
        if not path.is_dir():
            continue
        match = epoch_pattern.match(path.name)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, path))

    if not checkpoints:
        return

    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])

    # Last checkpoint is the one with highest epoch number
    last_epoch, last_dir = checkpoints[-1]

    # Determine best checkpoint by reading metrics and comparing val rewards
    best_epoch, best_dir = last_epoch, last_dir  # Default to last
    best_reward = float("-inf")

    for epoch, ckpt_dir in checkpoints:
        metrics_file = ckpt_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        from utils.io import read_json
        metrics = read_json(metrics_file)

        # Look for val reward metric (try multiple possible keys)
        val_reward = (
            metrics.get("val/roll/ep_rew/mean") or
            metrics.get("val/roll/ep_rew/best") or
            metrics.get("val_ep_rew_mean") or
            float("-inf")
        )

        if val_reward > best_reward:
            best_reward = val_reward
            best_epoch, best_dir = epoch, ckpt_dir

    # Create symlinks (remove existing ones first)
    last_symlink = checkpoints_dir / "@last"
    best_symlink = checkpoints_dir / "@best"

    if last_symlink.exists() or last_symlink.is_symlink():
        last_symlink.unlink()
    if best_symlink.exists() or best_symlink.is_symlink():
        best_symlink.unlink()

    # Create new symlinks (relative paths for portability)
    last_symlink.symlink_to(last_dir.name, target_is_directory=True)
    best_symlink.symlink_to(best_dir.name, target_is_directory=True)

    print(f"  Recreated symlinks: @last → {last_dir.name}, @best → {best_dir.name}")


def _infer_project_from_registry(run_id: str) -> Optional[str]:
    """Try to infer project from local runs registry."""
    try:
        from utils.run import _read_registry
        entries = _read_registry()
        for entry in entries:
            if entry.get("run_id") == run_id:
                return entry.get("project_id")
    except Exception:
        pass
    return None


def _search_wandb_for_run(run_id: str, entity: str) -> Optional[str]:
    """Search W&B for a run and return its project name."""
    try:
        api = wandb.Api()
        # Try to find the run across all projects
        # W&B API doesn't have a direct "search by run_id" across projects,
        # so we need to try common project names or iterate
        runs = api.runs(f"{entity}", filters={"display_name": run_id})
        for run in runs:
            if run.name == run_id:
                return run.project
    except Exception:
        pass
    return None


def download_run_artifact(
    run_id: str,
    entity: Optional[str] = None,
    project: Optional[str] = None,
    target_dir: Optional[Path] = None,
) -> Path:
    """Download a run archive artifact from W&B and extract it.

    Args:
        run_id: The run ID to download
        entity: W&B entity (username/org). Defaults to WANDB_ENTITY env var.
        project: W&B project name. Defaults to WANDB_PROJECT env var, or inferred from registry.
        target_dir: Directory to extract to. Defaults to "runs/".

    Returns:
        Path to the extracted run directory

    Raises:
        FileNotFoundError: If artifact not found in W&B
        ValueError: If entity or project cannot be determined
    """
    # Resolve entity and project
    entity = entity or os.getenv("WANDB_ENTITY")
    project = project or os.getenv("WANDB_PROJECT")

    if not entity:
        raise ValueError("entity must be provided or set via WANDB_ENTITY environment variable")

    # Try to infer project if not provided
    if not project:
        project = _infer_project_from_registry(run_id)
        if project:
            print(f"Inferred project from registry: {project}")

    if not project:
        project = _search_wandb_for_run(run_id, entity)
        if project:
            print(f"Found project via W&B search: {project}")

    if not project:
        raise ValueError(
            f"project must be provided or set via WANDB_PROJECT environment variable. "
            f"Could not infer project for run {run_id}. "
            f"Try setting WANDB_PROJECT in your .env file or pass it explicitly."
        )

    target_dir = target_dir or Path("runs")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B API
    api = wandb.Api()

    # Construct artifact name
    artifact_name = f"{entity}/{project}/run-{run_id}:latest"

    print(f"Downloading artifact: {artifact_name}")

    # Download artifact
    try:
        artifact = api.artifact(artifact_name, type="run-archive")
    except wandb.errors.CommError as e:
        raise FileNotFoundError(
            f"Artifact not found in W&B: {artifact_name}. "
            f"Ensure the run was uploaded via UploadRunCallback."
        ) from e

    # Download to temp directory
    artifact_dir = artifact.download()
    artifact_path = Path(artifact_dir)

    # Find the zip file
    zip_files = list(artifact_path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip file found in artifact: {artifact_path}")

    zip_path = zip_files[0]
    print(f"Extracting {zip_path.name} to {target_dir}")

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(target_dir)

    # Return path to extracted run directory
    run_dir = target_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Expected run directory not found after extraction: {run_dir}"
        )

    # Recreate checkpoint symlinks
    _recreate_checkpoint_symlinks(run_dir)

    print(f"✓ Run {run_id} downloaded and extracted to {run_dir}")
    return run_dir
