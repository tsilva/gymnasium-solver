"""Modal AI training runner for remote training execution.

Enables running training jobs on Modal infrastructure by specifying --backend modal in train.py.

Usage:
    # Run training remotely on Modal
    python train.py CartPole-v1:ppo --backend modal

    # With additional arguments
    python train.py CartPole-v1:ppo --backend modal --max-env-steps 50000 -q

    # Resume training remotely
    python train.py --resume @last --backend modal

Requirements:
    - Modal account (modal.com)
    - Modal token configured: `modal token new`
    - WANDB_API_KEY set as Modal secret (optional, for W&B logging)
"""

import os
import tomllib
from pathlib import Path

import modal

# Define Modal app
app = modal.App("gymnasium-solver-train")

# Get the project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).parent.parent


def get_training_dependencies():
    """Read dependencies from pyproject.toml and filter for training.

    Returns:
        List of dependency strings suitable for pip_install()
    """
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    all_deps = pyproject["project"]["dependencies"]

    # Dependencies to exclude from Modal image (not needed for training)
    exclude = {
        "modal",  # Modal SDK not needed inside containers
        "ipykernel",  # Jupyter kernel not needed for training
        "jupyter",  # Jupyter not needed for training
        "pytest",  # Testing framework not needed for training
        "gradio",  # UI framework not needed for training
        "matplotlib",  # Plotting not needed for training
        "python-dotenv",  # Env loading not needed (Modal handles secrets)
    }

    # Filter out excluded dependencies
    training_deps = []
    for dep in all_deps:
        # Extract package name (before any version specifiers or extras)
        pkg_name = (
            dep.split("[")[0]
            .split(">")[0]
            .split("<")[0]
            .split("=")[0]
            .split("!")[0]
            .strip()
        )
        if pkg_name not in exclude:
            training_deps.append(dep)

    return training_deps


# Create Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "swig",  # Required for box2d-py compilation
        "build-essential",  # C/C++ compiler for native extensions
        "libgl1-mesa-glx",  # OpenGL for rendering
        "libglib2.0-0",  # Required by some Gym environments
    )
    .pip_install(*get_training_dependencies())
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],  # Expects WANDB_API_KEY (optional)
    cpu=4.0,  # 4 CPUs per training job
    memory=8192,  # 8GB RAM
    timeout=7200,  # 2 hours max per run
)
def run_training(train_args: list[str]):
    """Run training with given CLI arguments on Modal.

    Args:
        train_args: List of CLI arguments to pass to train.py (excluding --modal)

    Yields:
        Log lines from training process
    """
    import subprocess
    import tempfile

    # Get repository URL from environment or use default
    repo_url = os.environ.get(
        "REPO_URL", "https://github.com/tsilva/gymnasium-solver.git"
    )

    # Clone repository to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        yield f"Cloning repository to {tmpdir}...\n"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmpdir],
            check=True,
            capture_output=True,
        )

        # Change to repo directory
        os.chdir(tmpdir)
        yield f"Working directory: {os.getcwd()}\n"

        # Set environment variables for quiet operation
        os.environ["VIBES_QUIET"] = "1"
        os.environ["VIBES_DISABLE_SESSION_LOGS"] = "1"

        # Build command
        cmd = ["python", "-u", "train.py"] + train_args  # -u for unbuffered output
        yield f"Running: {' '.join(cmd)}\n\n"

        # Run training with stdout/stderr capture for streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Stream output line by line
        for line in process.stdout:
            yield line

        # Wait for process to complete
        returncode = process.wait()

        if returncode != 0:
            yield f"\nTraining failed with return code {returncode}\n"
            raise RuntimeError("Training execution failed")

        yield "\nTraining completed successfully\n"


def launch_modal_training(args):
    """Launch training on Modal with given arguments.

    Args:
        args: Parsed argparse arguments from train.py
    """
    # Build CLI arguments list (exclude --modal flag)
    train_args = []

    # Add config spec (positional)
    if args.config:
        train_args.append(args.config)
    elif args.config_id:
        train_args.extend(["--config_id", args.config_id])

    # Add optional flags
    if args.wandb_sweep:
        train_args.append("--wandb_sweep")

    if args.max_env_steps:
        train_args.extend(["--max-env-steps", str(args.max_env_steps)])

    if hasattr(args, 'overrides') and args.overrides:
        for override in args.overrides:
            train_args.extend(["--override", override])

    if args.resume:
        train_args.extend(["--resume", args.resume])

    if args.epoch:
        train_args.extend(["--epoch", args.epoch])

    if hasattr(args, 'init_from_run') and args.init_from_run:
        train_args.extend(["--init-from-run", args.init_from_run])

    # Display info
    print("Launching training on Modal...")
    print(f"Arguments: {' '.join(train_args)}")
    print("\nView dashboard: https://modal.com/apps")
    print("\nStreaming logs...\n")
    print("-" * 80)

    # Run on Modal and stream logs via generator
    try:
        with app.run():
            for log_line in run_training.remote_gen(train_args):
                print(log_line, end="", flush=True)
        print("-" * 80)
        print("\n✓ Modal training completed!")
    except KeyboardInterrupt:
        print("\n✗ Modal training cancelled")
        raise
    except Exception as e:
        print("-" * 80)
        print(f"\n✗ Modal training failed: {e}")
        raise
