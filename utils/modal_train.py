"""Modal AI training runner for remote training execution.

Enables running training jobs on Modal infrastructure by specifying --backend modal in train.py.
Automatically allocates resources (CPU, memory, GPU, timeout) based on environment configuration.

Usage:
    # Run training remotely on Modal (resources auto-allocated)
    python train.py CartPole-v1:ppo --backend modal

    # With additional arguments
    python train.py CartPole-v1:ppo --backend modal --max-env-steps 50000 -q

    # Resume training remotely (uses default resources)
    python train.py --resume @last --backend modal

Resource Allocation:
    - CPU: Scaled based on n_envs (2-16 cores)
    - Memory: Scaled based on env type and n_envs (4-32GB)
    - GPU: Automatically allocated for image-based envs (T4 or A10G)
    - Timeout: Estimated based on max_env_steps (10 min - 4 hours)

Examples:
    - CartPole (vector): 4 CPUs, 6GB RAM, no GPU
    - Pong (image, 5M steps): 8 CPUs, 22GB RAM, T4 GPU
    - Pong (image, 15M steps): 8 CPUs, 24GB RAM, A10G GPU (upgraded)

Requirements:
    - Modal account (modal.com)
    - Modal token configured: `modal token new`
    - WANDB_API_KEY set as Modal secret (optional, for W&B logging)
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import modal

# Define Modal app
app = modal.App("gymnasium-solver-train")

# Get the project root (where pyproject.toml lives)
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ResourceRequirements:
    """Resource requirements for Modal training job."""

    cpu: float  # Number of CPU cores
    memory: int  # Memory in MB
    gpu: Optional[str]  # GPU type (e.g., "T4", "A10G", "A100") or None
    timeout: int  # Timeout in seconds

    def __str__(self) -> str:
        gpu_str = f", GPU: {self.gpu}" if self.gpu else ""
        return f"CPU: {self.cpu}, Memory: {self.memory}MB{gpu_str}, Timeout: {self.timeout}s"


def estimate_resources_from_config(config_id: str, variant_id: str, max_env_steps: Optional[int] = None) -> ResourceRequirements:
    """Estimate resource requirements from config.

    Args:
        config_id: Environment ID (e.g., "CartPole-v1")
        variant_id: Algorithm variant (e.g., "ppo")
        max_env_steps: Override for max_env_steps

    Returns:
        ResourceRequirements with estimated CPU, memory, GPU, and timeout
    """
    from utils.config import load_config

    # Load config to inspect parameters
    config = load_config(config_id, variant_id)

    # Override max_env_steps if provided
    if max_env_steps is not None:
        config.max_env_steps = max_env_steps

    # Determine if GPU is needed
    # Use GPU for:
    # 1. CNN policies (image-based)
    # 2. RGB/objects observation types
    # 3. Large environments
    policy_str = config.policy.value if hasattr(config.policy, "value") else str(config.policy)
    obs_type_str = config.obs_type.value if hasattr(config.obs_type, "value") else str(config.obs_type)

    needs_gpu = False
    gpu_type = None

    if "cnn" in policy_str or obs_type_str in {"rgb", "objects"}:
        needs_gpu = True
        # Use T4 for most tasks (cheapest GPU on Modal)
        # For very large image environments or long training, could use A10G
        if config.max_env_steps and config.max_env_steps > 10_000_000:
            gpu_type = "A10G"
        else:
            gpu_type = "T4"

    # Estimate CPU requirements based on n_envs
    # Rule of thumb: 1 CPU per 2-4 parallel environments, minimum 2, maximum 16
    n_envs = config.n_envs
    if needs_gpu:
        # With GPU, CPU is mainly for data loading and env stepping
        cpu = max(2.0, min(8.0, n_envs / 4.0))
    else:
        # Without GPU, CPU does everything
        cpu = max(2.0, min(16.0, n_envs / 2.0))

    # Estimate memory requirements
    # Base memory: 4GB
    # Add per environment: 100MB for vector envs, 500MB for image envs
    # Add for model size: 1GB for MLP, 2-4GB for CNN
    base_memory = 4096  # 4GB base

    if "cnn" in policy_str or obs_type_str in {"rgb", "objects"}:
        # Image-based
        per_env_memory = 500
        model_memory = 2048 if gpu_type == "T4" else 4096
    else:
        # Vector-based
        per_env_memory = 100
        model_memory = 1024

    total_memory = base_memory + (n_envs * per_env_memory) + model_memory

    # Estimate timeout based on max_env_steps
    # Rough estimate: 1000 env_steps per second for simple envs, 100 for complex
    if config.max_env_steps:
        if needs_gpu:
            # GPU training is faster for image envs
            steps_per_sec = 500 if obs_type_str in {"rgb", "objects"} else 2000
        else:
            # CPU training
            steps_per_sec = 2000 if obs_type_str == "vector" else 200

        # Add 50% buffer and minimum 10 minutes
        estimated_seconds = config.max_env_steps / steps_per_sec
        timeout = max(600, int(estimated_seconds * 1.5))
        # Cap at 4 hours
        timeout = min(timeout, 14400)
    else:
        # Default to 2 hours if max_env_steps not specified
        timeout = 7200

    return ResourceRequirements(
        cpu=cpu,
        memory=total_memory,
        gpu=gpu_type,
        timeout=timeout,
    )


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


def create_training_function(resources: ResourceRequirements):
    """Create a Modal function with specified resource requirements.

    Args:
        resources: ResourceRequirements specifying CPU, memory, GPU, and timeout

    Returns:
        Modal function configured with the specified resources
    """
    # Build function kwargs
    function_kwargs = {
        "image": image,
        "secrets": [modal.Secret.from_name("wandb-secret")],
        "cpu": resources.cpu,
        "memory": resources.memory,
        "timeout": resources.timeout,
    }

    # Add GPU if required
    if resources.gpu:
        # Modal now accepts GPU as string (e.g., "T4", "A10G", "A100")
        function_kwargs["gpu"] = resources.gpu

    @app.function(serialized=True, **function_kwargs)
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

    return run_training


def launch_modal_training(args):
    """Launch training on Modal with given arguments.

    Args:
        args: Parsed argparse arguments from train.py
    """
    # Parse config_id and variant_id to estimate resources
    config_spec = None
    if args.config:
        config_spec = args.config
    elif args.config_id:
        config_spec = args.config_id

    # For resume mode, we can't easily estimate resources without the original config
    # Use default resources in that case
    if args.resume or not config_spec:
        print("Using default resource allocation (resume mode or missing config)")
        resources = ResourceRequirements(
            cpu=4.0,
            memory=8192,
            gpu=None,
            timeout=7200,
        )
    else:
        # Parse env:variant format
        if ":" in config_spec:
            config_id, variant_id = config_spec.split(":", 1)
        else:
            raise ValueError(
                f"Config spec must be in 'env:variant' format (e.g., 'CartPole-v1:ppo'), got: {config_spec}"
            )

        # Estimate resources from config
        print(f"Estimating resource requirements for {config_id}:{variant_id}...")
        resources = estimate_resources_from_config(
            config_id=config_id,
            variant_id=variant_id,
            max_env_steps=args.max_env_steps,
        )
        print(f"Resource requirements: {resources}")

    # Build CLI arguments list (exclude --backend flag)
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
    print("\nLaunching training on Modal...")
    print(f"Arguments: {' '.join(train_args)}")
    print("\nView dashboard: https://modal.com/apps")
    print("\nStreaming logs...\n")
    print("-" * 80)

    # Create Modal function with dynamic resources
    run_training_fn = create_training_function(resources)

    # Run on Modal and stream logs via generator
    try:
        with app.run():
            for log_line in run_training_fn.remote_gen(train_args):
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
