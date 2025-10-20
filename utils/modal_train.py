"""Modal AI training runner for remote training execution.

Enables running training jobs on Modal infrastructure by specifying --backend modal in train.py.
Automatically allocates resources (CPU, memory, GPU, timeout) based on environment configuration.

Usage:
    # Run training remotely on Modal (resources auto-allocated, streams logs)
    python train.py CartPole-v1:ppo --backend modal

    # Run in detached mode (job continues if terminal is closed)
    python train.py CartPole-v1:ppo --backend modal --detach

    # With additional arguments
    python train.py CartPole-v1:ppo --backend modal --max-env-steps 50000 -q

    # Resume training remotely (uses default resources)
    python train.py --resume @last --backend modal

Execution Modes:
    - Streaming (default): Streams logs to terminal, blocks until completion. Job stops if terminal is killed.
    - Detached (--detach): Spawns job and returns immediately. Job continues running even if terminal is closed.
      Monitor via Modal dashboard at https://modal.com/apps or W&B.

Modal App Naming:
    - Apps are named with the pattern: gymnasium-solver-{project_id}-{run_id}
    - Example: gymnasium-solver-CartPole-v1-abc123xyz
    - For resume mode: gymnasium-solver-unknown-resume (until run ID is determined from checkpoint)

Resource Allocation:
    - CPU: Scaled based on n_envs (2-16 cores)
    - Memory: Scaled based on env type and n_envs (4-32GB)
    - GPU: Automatically allocated for image-based envs (T4 or A10G)
    - Timeout: Estimated based on max_env_steps (10 min - 4 hours)

Preemption Handling:
    - Modal Functions are subject to preemption (rare but possible)
    - When preempted, SIGTERM is sent to allow graceful shutdown
    - Training process saves checkpoint and uploads to W&B before exit
    - On restart (after preemption), training automatically resumes from last checkpoint
    - Implementation:
        1. Signal handler forwards SIGTERM to training subprocess
        2. Before training starts, checks W&B for existing checkpoints
        3. If found, automatically converts to resume mode

Examples:
    - CartPole (vector): 4 CPUs, 6GB RAM, no GPU
    - Pong (image, 5M steps): 8 CPUs, 22GB RAM, T4 GPU
    - Pong (image, 15M steps): 8 CPUs, 24GB RAM, A10G GPU (upgraded)

Requirements:
    - Modal account (modal.com)
    - Modal token configured: `modal token new`
    - WANDB_API_KEY and WANDB_ENTITY set as Modal secret (for W&B logging and preemption recovery)
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import modal

# Volume for Retro ROMs
roms_volume = modal.Volume.from_name("roms", create_if_missing=True)

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
        config.max_env_steps = int(max_env_steps) if isinstance(max_env_steps, str) else max_env_steps

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
    # Rule of thumb: 1 CPU per 2 parallel environments, minimum 2, maximum 16
    n_envs = config.n_envs
    if needs_gpu:
        # With GPU, CPU is mainly for data loading and env stepping
        cpu = max(2.0, min(16.0, n_envs / 2.0))
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
        # Ensure max_env_steps is an int (may be string from YAML)
        max_steps = int(config.max_env_steps) if isinstance(config.max_env_steps, str) else config.max_env_steps

        if needs_gpu:
            # GPU training is faster for image envs
            steps_per_sec = 500 if obs_type_str in {"rgb", "objects"} else 2000
        else:
            # CPU training
            steps_per_sec = 2000 if obs_type_str == "vector" else 200

        # Add 50% buffer and minimum 10 minutes
        estimated_seconds = max_steps / steps_per_sec
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
    .env({"PYTHONPATH": "/root/gymnasium-solver"})  # Make local modules importable during deserialization
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/gymnasium-solver",
        # Exclude unnecessary directories and files to reduce image size and avoid build conflicts
        ignore=[
            ".git",
            "__pycache__",
            ".pytest_cache",
            "runs",
            ".venv",
            "venv",
            "node_modules",
            "TODO.md",  # Exclude TODO to avoid conflicts during build
            "*.log",  # Exclude log files
            ".DS_Store",  # Exclude macOS system files
        ],
    )
)


def _check_run_has_checkpoints_in_wandb(run_id: str, project_id: str) -> bool:
    """Check if a run exists in W&B with checkpoints.

    Args:
        run_id: Run ID to check
        project_id: W&B project ID

    Returns:
        True if run exists with checkpoints in W&B, False otherwise
    """
    try:
        import wandb

        # Get W&B entity from environment (set in Modal secrets)
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            return False

        # Initialize W&B API
        api = wandb.Api()

        # Check if run exists
        try:
            run = api.run(f"{entity}/{project_id}/{run_id}")
        except wandb.errors.CommError:
            return False

        # Check for run-archive artifact
        artifact_name = f"run-{run_id}"
        try:
            # List artifacts for this run
            artifacts = list(run.logged_artifacts())
            for artifact in artifacts:
                if artifact.name.startswith(artifact_name) and artifact.type == "run-archive":
                    # Found the run archive, assume it has checkpoints
                    return True
        except:
            pass

        return False
    except Exception:
        # If any error occurs, assume no checkpoints (fail safe)
        return False


def create_training_function(app: modal.App, resources: ResourceRequirements, detached: bool = False):
    """Create a Modal function with specified resource requirements.

    Args:
        app: Modal app to attach the function to
        resources: ResourceRequirements specifying CPU, memory, GPU, and timeout
        detached: If True, create non-generator function for spawn(). If False, create generator for remote_gen()

    Returns:
        Modal function configured with the specified resources
    """
    # Build function kwargs
    # Note: wandb-secret should contain WANDB_API_KEY and optionally WANDB_ENTITY
    # WANDB_ENTITY is used for cross-project artifact downloads during transfer learning
    function_kwargs = {
        "image": image,
        "secrets": [modal.Secret.from_name("wandb-secret")],
        "volumes": {"/roms": roms_volume},
        "cpu": resources.cpu,
        "memory": resources.memory,
        "timeout": resources.timeout,
    }

    # Add GPU if required
    if resources.gpu:
        # Modal now accepts GPU as string (e.g., "T4", "A10G", "A100")
        function_kwargs["gpu"] = resources.gpu

    if detached:
        # Non-generator version for detached execution via spawn()
        @app.function(serialized=True, **function_kwargs)
        def run_training(train_args: list[str], run_id: str = None, project_id: str = None):
            """Run training with given CLI arguments on Modal.

            Args:
                train_args: List of CLI arguments to pass to train.py (excluding --modal)
                run_id: Pre-generated run ID to use for W&B run naming
                project_id: W&B project ID from config (used to set WANDB_PROJECT env var)

            Returns:
                Exit code (0 for success, non-zero for failure)
            """
            import signal
            import subprocess

            # Change to mounted code directory
            code_dir = "/root/gymnasium-solver"
            os.chdir(code_dir)
            print(f"Working directory: {os.getcwd()}", flush=True)

            # Set environment variables for quiet operation
            os.environ["VIBES_QUIET"] = "1"
            os.environ["VIBES_DISABLE_SESSION_LOGS"] = "1"

            # Suppress warnings in all subprocesses (env stepping, multiprocessing)
            # This catches warnings that happen in forked/spawned worker processes
            os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pydantic._internal._generate_schema,ignore::UserWarning:torch"

            # If project_id was provided, set it as environment variable
            # This is needed for W&B artifact downloads during --init-from-run
            if project_id:
                os.environ["WANDB_PROJECT"] = project_id
                print(f"Using W&B project: {project_id}", flush=True)

            # If run_id was provided, set it as environment variable
            # so train.py will use it for W&B initialization
            if run_id:
                os.environ["WANDB_RUN_ID"] = run_id
                print(f"Using pre-generated run ID: {run_id}", flush=True)

            # Preemption handling: Check if this is a resumed run
            # If run_id exists in W&B with checkpoints, auto-resume instead of starting fresh
            is_resuming = "--resume" in train_args
            if run_id and project_id and not is_resuming:
                print(f"Checking W&B for existing checkpoints...", flush=True)
                has_checkpoints = _check_run_has_checkpoints_in_wandb(run_id, project_id)
                if has_checkpoints:
                    print(f"⚠ Detected preemption restart: run {run_id} has existing checkpoints in W&B", flush=True)
                    print(f"Automatically resuming from last checkpoint...", flush=True)
                    # Convert to resume mode
                    train_args = ["--resume", run_id] + [arg for arg in train_args if not arg.startswith("--")]
                    is_resuming = True
                else:
                    print(f"No existing checkpoints found, starting fresh training", flush=True)

            # Check if this is a Retro environment and import ROM if needed
            config_spec = None
            for i, arg in enumerate(train_args):
                if not arg.startswith("--") and ":" in arg:
                    config_spec = arg
                    break
                elif arg == "--config_id" and i + 1 < len(train_args):
                    config_spec = train_args[i + 1]
                    break

            if config_spec and config_spec.startswith("Retro"):
                # Extract game ID from config spec (e.g., "Retro-SuperMarioBros-Nes:ppo" -> "SuperMarioBros-Nes")
                env_id = config_spec.split(":")[0]  # Get env part before variant
                rom_game_id = env_id.replace("Retro-", "")  # Remove "Retro-" prefix

                print(f"Detected Retro environment: {rom_game_id}", flush=True)
                print(f"Importing ROM from volume...", flush=True)

                rom_path = Path(f"/roms/retro-roms/{rom_game_id}")
                if not rom_path.exists():
                    raise RuntimeError(
                        f"ROM directory not found in volume at {rom_path}. "
                        f"Upload ROM with: python scripts/upload_rom_to_modal.py {rom_game_id}"
                    )

                # Import the ROM using retro.import
                import_cmd = ["python", "-m", "retro.import", str(rom_path)]
                print(f"Running: {' '.join(import_cmd)}", flush=True)
                result = subprocess.run(import_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"ROM import failed:", flush=True)
                    print(f"STDOUT: {result.stdout}", flush=True)
                    print(f"STDERR: {result.stderr}", flush=True)
                    raise RuntimeError(f"ROM import failed with return code {result.returncode}")

                print(f"ROM imported successfully\n", flush=True)

            # Build command
            cmd = ["python", "-u", "train.py"] + train_args  # -u for unbuffered output
            print(f"Running: {' '.join(cmd)}\n", flush=True)

            # Run training with stdout/stderr capture for streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Signal handler to forward signals to subprocess (for preemption handling)
            def forward_signal(signum, frame):
                print(f"\n⚠ Received signal {signum}, forwarding to training process...", flush=True)
                if process.poll() is None:  # Process still running
                    process.send_signal(signum)

            # Register signal handlers for graceful shutdown on preemption
            # In detached mode, only handle SIGTERM (preemption), ignore SIGINT
            # SIGINT can arrive spuriously from Modal infrastructure and shouldn't kill training
            signal.signal(signal.SIGTERM, forward_signal)

            # Stream output line by line
            for line in process.stdout:
                print(line, end="", flush=True)

            # Wait for process to complete
            returncode = process.wait()

            if returncode != 0:
                print(f"\nTraining failed with return code {returncode}", flush=True)
                raise RuntimeError("Training execution failed")

            print("\nTraining completed successfully", flush=True)
            return returncode
    else:
        # Generator version for streaming execution via remote_gen()
        @app.function(serialized=True, **function_kwargs)
        def run_training(train_args: list[str], run_id: str = None, project_id: str = None):
            """Run training with given CLI arguments on Modal.

            Args:
                train_args: List of CLI arguments to pass to train.py (excluding --modal)
                run_id: Pre-generated run ID to use for W&B run naming
                project_id: W&B project ID from config (used to set WANDB_PROJECT env var)

            Yields:
                Log lines from training process
            """
            import signal
            import subprocess

            def log_and_yield(msg: str):
                """Print to Modal container stdout and yield to local client."""
                print(msg, end="", flush=True)
                yield msg

            # Change to mounted code directory
            code_dir = "/root/gymnasium-solver"
            os.chdir(code_dir)
            yield from log_and_yield(f"Working directory: {os.getcwd()}\n")

            # Set environment variables for quiet operation
            os.environ["VIBES_QUIET"] = "1"
            os.environ["VIBES_DISABLE_SESSION_LOGS"] = "1"

            # Suppress warnings in all subprocesses (env stepping, multiprocessing)
            # This catches warnings that happen in forked/spawned worker processes
            os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:pydantic._internal._generate_schema,ignore::UserWarning:torch"

            # If project_id was provided, set it as environment variable
            # This is needed for W&B artifact downloads during --init-from-run
            if project_id:
                os.environ["WANDB_PROJECT"] = project_id
                yield from log_and_yield(f"Using W&B project: {project_id}\n")

            # If run_id was provided, set it as environment variable
            # so train.py will use it for W&B initialization
            if run_id:
                os.environ["WANDB_RUN_ID"] = run_id
                yield from log_and_yield(f"Using pre-generated run ID: {run_id}\n")

            # Preemption handling: Check if this is a resumed run
            # If run_id exists in W&B with checkpoints, auto-resume instead of starting fresh
            is_resuming = "--resume" in train_args
            if run_id and project_id and not is_resuming:
                yield from log_and_yield(f"Checking W&B for existing checkpoints...\n")
                has_checkpoints = _check_run_has_checkpoints_in_wandb(run_id, project_id)
                if has_checkpoints:
                    yield from log_and_yield(f"⚠ Detected preemption restart: run {run_id} has existing checkpoints in W&B\n")
                    yield from log_and_yield(f"Automatically resuming from last checkpoint...\n")
                    # Convert to resume mode
                    train_args = ["--resume", run_id] + [arg for arg in train_args if not arg.startswith("--")]
                    is_resuming = True
                else:
                    yield from log_and_yield(f"No existing checkpoints found, starting fresh training\n")

            # Check if this is a Retro environment and import ROM if needed
            config_spec = None
            for i, arg in enumerate(train_args):
                if not arg.startswith("--") and ":" in arg:
                    config_spec = arg
                    break
                elif arg == "--config_id" and i + 1 < len(train_args):
                    config_spec = train_args[i + 1]
                    break

            if config_spec and config_spec.startswith("Retro"):
                # Extract game ID from config spec (e.g., "Retro-SuperMarioBros-Nes:ppo" -> "SuperMarioBros-Nes")
                env_id = config_spec.split(":")[0]  # Get env part before variant
                rom_game_id = env_id.replace("Retro-", "")  # Remove "Retro-" prefix

                yield from log_and_yield(f"Detected Retro environment: {rom_game_id}\n")
                yield from log_and_yield(f"Importing ROM from volume...\n")

                rom_path = Path(f"/roms/retro-roms/{rom_game_id}")
                if not rom_path.exists():
                    raise RuntimeError(
                        f"ROM directory not found in volume at {rom_path}. "
                        f"Upload ROM with: python scripts/upload_rom_to_modal.py {rom_game_id}"
                    )

                # Import the ROM using retro.import
                import_cmd = ["python", "-m", "retro.import", str(rom_path)]
                yield from log_and_yield(f"Running: {' '.join(import_cmd)}\n")
                result = subprocess.run(import_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    yield from log_and_yield(f"ROM import failed:\n")
                    yield from log_and_yield(f"STDOUT: {result.stdout}\n")
                    yield from log_and_yield(f"STDERR: {result.stderr}\n")
                    raise RuntimeError(f"ROM import failed with return code {result.returncode}")

                yield from log_and_yield(f"ROM imported successfully\n\n")

            # Build command
            cmd = ["python", "-u", "train.py"] + train_args  # -u for unbuffered output
            yield from log_and_yield(f"Running: {' '.join(cmd)}\n\n")

            # Run training with stdout/stderr capture for streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Signal handler to forward signals to subprocess (for preemption handling)
            def forward_signal(signum, frame):
                msg = f"\n⚠ Received signal {signum}, forwarding to training process...\n"
                print(msg, end="", flush=True)
                if process.poll() is None:  # Process still running
                    process.send_signal(signum)

            # Register signal handlers for graceful shutdown on preemption
            # Only handle SIGTERM (preemption). In streaming mode, SIGINT is handled by Modal client.
            signal.signal(signal.SIGTERM, forward_signal)

            # Stream output line by line
            for line in process.stdout:
                yield from log_and_yield(line)

            # Wait for process to complete
            returncode = process.wait()

            if returncode != 0:
                yield from log_and_yield(f"\nTraining failed with return code {returncode}\n")
                raise RuntimeError("Training execution failed")

            yield from log_and_yield("\nTraining completed successfully\n")

    return run_training


def launch_modal_training(args):
    """Launch training on Modal with given arguments.

    Args:
        args: Parsed argparse arguments from train.py
    """
    import wandb
    from utils.config import load_config

    # Check if detached mode is requested
    detached = getattr(args, 'detach', False)

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
        run_id = None  # Resume mode will use existing run ID
        project_id = "unknown"  # Will be determined from checkpoint
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

        # Generate run ID locally for W&B and Modal app naming
        # Note: We do NOT create a local run directory for Modal runs
        # since all artifacts are stored remotely in W&B
        run_id = wandb.util.generate_id()
        config = load_config(config_id, variant_id)

        # Extract project_id from config to set WANDB_PROJECT in Modal
        # This is needed for W&B artifact downloads (e.g., --init-from-run)
        project_id = config.project_id

    # Create Modal app with dynamic name
    app_name = f"gymnasium-solver-{project_id}-{run_id or 'resume'}"
    print(f"Modal app name: {app_name}")
    app = modal.App(app_name)

    # Build CLI arguments list (exclude --backend and --detach flags)
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
    print(f"Mode: {'Detached' if detached else 'Streaming (attached)'}")
    print("\nView dashboard: https://modal.com/apps")

    # Create Modal function with dynamic resources
    run_training_fn = create_training_function(app, resources, detached=detached)

    if detached:
        # Detached mode: spawn the function and return immediately
        print("\nSpawning detached training job...")
        print("-" * 80)
        try:
            # Spawn function inside app context
            with app.run():
                function_call = run_training_fn.spawn(train_args, run_id, project_id)
                print(f"\nFunction call ID: {function_call.object_id}")

                # CRITICAL: Wait for Modal to schedule the function before exiting context
                # If we exit too quickly, Modal cancels the pending function
                # Give Modal time to transition from queued -> running state
                import time
                print("Waiting for Modal to schedule the function...")
                time.sleep(5)  # 5 seconds should be sufficient for Modal to start the function

            # Context exited - function is now running independently on Modal
        except KeyboardInterrupt:
            # User interrupted during spawn setup
            print("\n✗ Job spawn cancelled by user")
            raise
        except Exception as e:
            print("-" * 80)
            print(f"\n✗ Failed to spawn training job: {e}")
            raise

        # Spawn succeeded - job will continue running even if terminal is closed
        print("\n✓ Training job spawned successfully!")
        print(f"Run ID: {run_id or '(determined from checkpoint)'}")
        print("\nThe training will continue running on Modal even if you close this terminal.")
        print("Monitor progress at: https://modal.com/apps")
        if run_id:
            print(f"Or check W&B: https://wandb.ai")
    else:
        # Streaming mode: stream logs and block until completion (original behavior)
        print("\nStreaming logs...\n")
        print("-" * 80)
        try:
            with app.run():
                for log_line in run_training_fn.remote_gen(train_args, run_id, project_id):
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
