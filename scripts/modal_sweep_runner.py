"""Modal AI sweep runner for W&B hyperparameter sweeps.

Scales out W&B sweep agents across Modal CPU instances for parallel training.

Usage:
    # Create sweep first (returns sweep_id)
    wandb sweep config/sweeps/cartpole_ppo_grid.yaml

    # Launch Modal workers
    modal run scripts/modal_sweep_runner.py --sweep-id <sweep_id> --count 10

    # Or use entity/project explicitly
    modal run scripts/modal_sweep_runner.py --sweep-id <sweep_id> --entity <entity> --project <project> --count 10

Requirements:
    - Modal account (modal.com)
    - Modal token configured: `modal token new`
    - WANDB_API_KEY set as Modal secret
"""

import os

import modal

# Define Modal app
app = modal.App("wandb-sweep-runner")

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
    .pip_install(
        # Core dependencies (match pyproject.toml)
        "gymnasium[classic-control,atari,box2d,mujoco,other]",
        "gymnasium-robotics",
        "ale-py",
        "minari[hdf5]",
        "torch",
        "numpy",
        "wandb",
        "wandb-workspaces",
        "ocatari>=2.2.1",
        "pytorch-lightning",
        "torchvision",
        "pyyaml",
        "ruamel.yaml",
        "watchdog",
        "setuptools<80",
        "huggingface_hub>=0.22.0",
        "vizdoom",
        "mcp",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],  # Expects WANDB_API_KEY
    cpu=2.0,  # 2 CPUs per worker
    memory=4096,  # 4GB RAM
    timeout=3600,  # 1 hour max per run
)
def run_sweep_agent(sweep_id: str, entity: str, project: str, count: int = 1):
    """Run W&B sweep agent for given sweep.

    Args:
        sweep_id: W&B sweep ID (e.g., 'abc123xyz')
        entity: W&B entity/username
        project: W&B project name
        count: Number of runs this agent should execute (default: 1)
    """
    import subprocess
    import tempfile

    # Get repository URL from environment or use default
    repo_url = os.environ.get(
        "REPO_URL", "https://github.com/tsilva/gymnasium-solver.git"
    )

    # Clone repository to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning repository to {tmpdir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmpdir],
            check=True,
            capture_output=True,
        )

        # Change to repo directory
        os.chdir(tmpdir)
        print(f"Working directory: {os.getcwd()}")

        # Set environment variables for quiet operation
        os.environ["VIBES_QUIET"] = "1"
        os.environ["VIBES_DISABLE_SESSION_LOGS"] = "1"

        # Verify wandb is available and logged in
        result = subprocess.run(
            ["python", "-c", "import wandb; print(wandb.__version__)"],
            capture_output=True,
            text=True,
        )
        print(f"W&B version: {result.stdout.strip()}")

        # Construct sweep path
        sweep_path = f"{entity}/{project}/{sweep_id}"
        print(f"Starting sweep agent for: {sweep_path}")
        print(f"Will execute {count} run(s)")

        # Run sweep agent
        # Use subprocess to run wandb agent command
        cmd = ["wandb", "agent", "--count", str(count), sweep_path]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"Sweep agent failed with return code {result.returncode}")
            raise RuntimeError(f"Sweep agent execution failed")

        print("Sweep agent completed successfully")


@app.local_entrypoint()
def main(
    sweep_id: str,
    entity: str = None,
    project: str = None,
    count: int = 10,
    runs_per_worker: int = 1,
):
    """Launch multiple Modal workers to run W&B sweep.

    Args:
        sweep_id: W&B sweep ID (e.g., 'abc123xyz')
        entity: W&B entity/username (default: from WANDB_ENTITY env var)
        project: W&B project name (default: from WANDB_PROJECT env var)
        count: Total number of sweep runs to execute across all workers (default: 10)
        runs_per_worker: Number of runs each worker should execute (default: 1)
    """
    # Get entity/project from environment if not provided
    entity = entity or os.environ.get("WANDB_ENTITY")
    project = project or os.environ.get("WANDB_PROJECT")

    if not entity:
        raise ValueError(
            "entity must be provided via --entity flag or WANDB_ENTITY environment variable"
        )
    if not project:
        raise ValueError(
            "project must be provided via --project flag or WANDB_PROJECT environment variable"
        )

    print(f"Launching sweep: {entity}/{project}/{sweep_id}")
    print(f"Total runs: {count}")
    print(f"Runs per worker: {runs_per_worker}")

    # Calculate number of workers needed
    num_workers = (count + runs_per_worker - 1) // runs_per_worker
    print(f"Launching {num_workers} workers...")

    # Launch workers in parallel
    tasks = []
    for i in range(num_workers):
        # Calculate runs for this worker (handle remainder in last worker)
        worker_runs = min(runs_per_worker, count - (i * runs_per_worker))
        if worker_runs <= 0:
            break

        print(f"Worker {i+1}/{num_workers}: {worker_runs} run(s)")
        task = run_sweep_agent.spawn(sweep_id, entity, project, worker_runs)
        tasks.append(task)

    # Wait for all workers to complete
    print(f"\nWaiting for {len(tasks)} workers to complete...")
    for i, task in enumerate(tasks):
        try:
            task.get()
            print(f"✓ Worker {i+1}/{len(tasks)} completed")
        except Exception as e:
            print(f"✗ Worker {i+1}/{len(tasks)} failed: {e}")

    print("\nAll workers completed!")
