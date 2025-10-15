#!/usr/bin/env python3
"""Render checkpoint progression video showing policy evolution through training.

This script:
1. Loads all checkpoints from a training run (up to best epoch)
2. For each checkpoint, runs a single episode and records frames
3. Creates separator frames showing epoch numbers
4. Concatenates all videos with separators into a final progression video
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.config import Config
from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.random import set_random_seed
from utils.rollouts import RolloutCollector
from utils.run import Run


def parse_epoch_from_checkpoint(checkpoint_name: str) -> int | None:
    """Extract epoch number from checkpoint name like 'epoch=05'."""
    match = re.match(r"epoch=(\d+)", checkpoint_name)
    if match:
        return int(match.group(1))
    return None


def get_best_epoch(run: Run) -> int:
    """Resolve the best checkpoint epoch by following the symlink."""
    best_checkpoint_dir = run.best_checkpoint_dir
    if not best_checkpoint_dir.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {best_checkpoint_dir}")

    # Resolve the symlink to get the actual checkpoint directory
    resolved = best_checkpoint_dir.resolve()
    epoch = parse_epoch_from_checkpoint(resolved.name)

    if epoch is None:
        raise ValueError(f"Could not parse epoch from best checkpoint: {resolved.name}")

    return epoch


def list_checkpoints_up_to_best(run: Run, best_epoch: int) -> list[tuple[int, Path]]:
    """List all checkpoint directories up to and including best epoch, sorted by epoch."""
    checkpoints = []

    for checkpoint_dir in run.checkpoints_dir.iterdir():
        # Skip symlinks and non-directories
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            continue

        epoch = parse_epoch_from_checkpoint(checkpoint_dir.name)
        if epoch is not None and epoch <= best_epoch:
            checkpoints.append((epoch, checkpoint_dir))

    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def create_separator_frame(width: int, height: int, epoch: int, fps: int) -> ImageSequenceClip:
    """Create a 0.5-second separator clip showing the epoch number."""
    # Create a text clip
    text = f"Epoch {epoch}"
    duration = 0.5  # 0.5 seconds

    txt_clip = TextClip(
        text=text,
        font_size=min(width, height) // 10,
        color='white',
        bg_color='black',
        size=(width, height),
        method='label',
        horizontal_align='center',
        vertical_align='center'
    ).with_duration(duration)

    # Convert to ImageSequenceClip for consistency
    # Generate frames at the target FPS
    n_frames = int(duration * fps)
    frames = [txt_clip.get_frame(t) for t in np.linspace(0, duration, n_frames, endpoint=False)]

    return ImageSequenceClip(frames, fps=fps)


def _render_checkpoint_worker(args: tuple) -> tuple[int, Path]:
    """Worker function for parallel checkpoint rendering.

    Args:
        args: Tuple of (epoch, checkpoint_path, config, seed, video_path)

    Returns:
        Tuple of (epoch, video_path) for sorting and concatenation
    """
    epoch, checkpoint_path, config, seed, video_path = args

    # Render the episode
    render_episode_for_checkpoint(checkpoint_path, config, seed, video_path)

    return (epoch, video_path)


def render_episode_for_checkpoint(
    checkpoint_path: Path,
    config: Config,
    seed: int,
    video_path: Path
) -> None:
    """Render a single episode using the checkpoint policy and save to video."""
    epoch = parse_epoch_from_checkpoint(checkpoint_path.parent.name)
    print(f"  [PID {mp.current_process().pid}] Rendering episode from {checkpoint_path.parent.name}...")

    # Build environment with rgb_array mode
    env = build_env_from_config(
        config,
        n_envs=1,
        vectorization_mode='sync',
        render_mode='rgb_array',
        seed=seed
    )

    # Load policy model from checkpoint
    policy_model, _ = load_policy_model_from_checkpoint(checkpoint_path, env, config)

    # Create rollout collector
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=1,
        **config.rollout_collector_hyperparams(),
    )

    # Find the underlying SyncVectorEnv to access individual envs
    import gymnasium as gym
    current_env = env
    sync_vec_env = None
    while current_env is not None:
        if isinstance(current_env, gym.vector.SyncVectorEnv):
            sync_vec_env = current_env
            break
        current_env = getattr(current_env, 'env', None)

    assert sync_vec_env is not None, "Could not find SyncVectorEnv in wrapper chain"
    assert len(sync_vec_env.envs) > 0, "SyncVectorEnv has no environments"
    first_env = sync_vec_env.envs[0]

    # Collect frames for one complete episode
    frames = []
    episodes_finished = 0

    # Initial frame after reset
    env.reset()
    frame = first_env.render()
    frames.append(frame)

    while episodes_finished < 1:
        # Collect one step
        _ = collector.collect(deterministic=True)

        # Capture frame
        frame = first_env.render()
        frames.append(frame)

        # Check if episode finished
        finished_eps = collector.pop_recent_episodes()
        if finished_eps:
            episodes_finished += len(finished_eps)

    env.close()

    # Save frames as video
    if len(frames) > 0:
        fps = config.spec.get('render_fps', 30) if config.spec else 30
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(video_path), audio=False, logger=None)
        print(f"  Saved video: {video_path} ({len(frames)} frames)")
    else:
        raise RuntimeError("No frames were captured during episode rendering")


def main():
    parser = argparse.ArgumentParser(
        description="Render checkpoint progression video showing policy evolution"
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        default="@last",
        help="Run ID (default: @last)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default: runs/<run_id>/progression.mp4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for episodes (default: test seed from config)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for rendering (default: cpu_count)"
    )
    args = parser.parse_args()

    # Resolve run ID
    run_id = args.run_id
    if run_id == "@last":
        from utils.run import LAST_RUN_DIR
        if not LAST_RUN_DIR.exists():
            raise FileNotFoundError("No @last run found. Train a model first.")
        run_id = LAST_RUN_DIR.resolve().name

    # Load run and config
    print(f"Loading run: {run_id}")
    run = Run.load(run_id)
    config = run.load_config()

    # Determine seed
    seed = args.seed if args.seed is not None else config.seed_test
    set_random_seed(seed)

    # Get best epoch
    best_epoch = get_best_epoch(run)
    print(f"Best epoch: {best_epoch}")

    # List all checkpoints up to best
    checkpoints = list_checkpoints_up_to_best(run, best_epoch)
    print(f"Found {len(checkpoints)} checkpoints to render (epochs 0-{best_epoch})")

    if not checkpoints:
        raise ValueError("No checkpoints found to render")

    # Create temp directory for individual videos
    temp_dir = Path(tempfile.mkdtemp(prefix="checkpoint_progression_"))
    print(f"Using temp directory: {temp_dir}")

    # Determine number of workers
    n_workers = args.workers if args.workers is not None else mp.cpu_count()
    print(f"Using {n_workers} parallel workers for rendering")

    try:
        # Prepare tasks for parallel rendering
        tasks = []

        for epoch, checkpoint_dir in checkpoints:
            # Construct checkpoint path
            checkpoint_path = checkpoint_dir / "model.pt"
            if not checkpoint_path.exists():
                # Try old format
                checkpoint_path = checkpoint_dir / "policy.ckpt"

            if not checkpoint_path.exists():
                print(f"  WARNING: Checkpoint not found at {checkpoint_path}, skipping")
                continue

            # Prepare task
            video_path = temp_dir / f"epoch_{epoch:02d}.mp4"
            tasks.append((epoch, checkpoint_path, config, seed, video_path))

        # Render checkpoints in parallel
        print(f"Rendering {len(tasks)} checkpoints in parallel...")
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_render_checkpoint_worker, tasks)

        # Sort results by epoch
        results.sort(key=lambda x: x[0])

        # Load video clips and create separators
        fps = config.spec.get('render_fps', 30) if config.spec else 30
        frame_width = None
        frame_height = None
        video_clips = []

        for epoch, video_path in results:
            # Load the video clip
            clip = VideoFileClip(str(video_path))

            # Get frame dimensions from first video
            if frame_width is None:
                frame_width = clip.w
                frame_height = clip.h

            # Create separator before this epoch's video
            separator = create_separator_frame(frame_width, frame_height, epoch, fps)

            video_clips.append(separator)
            video_clips.append(clip)

        # Concatenate all clips
        print(f"\nConcatenating {len(video_clips)} clips...")
        final_clip = concatenate_videoclips(video_clips, method="compose")

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = run.run_dir / "progression.mp4"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write final video
        print(f"Writing final video to: {output_path}")
        final_clip.write_videofile(str(output_path), audio=False, logger=None)

        print(f"\nDone! Progression video saved to: {output_path}")

    finally:
        # Cleanup temp directory
        print(f"Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
