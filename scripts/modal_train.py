"""Modal AI runner for single training runs.

Usage:
    modal run scripts/modal_train.py --config-id Retro-SuperMarioBros-Nes:ppo --max-env-steps 1000000
"""

import os
import sys
from pathlib import Path

import modal

# Define Modal app
app = modal.App("gymnasium-solver-train")

# Get repo URL from environment or use default
REPO_URL = os.environ.get(
    "REPO_URL", "https://github.com/tsilva/gymnasium-solver.git"
)

# Create Modal image with all dependencies from pyproject.toml
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "swig",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    .run_commands(
        f"git clone --depth 1 {REPO_URL} /tmp/gymnasium-solver",
        "pip install -e /tmp/gymnasium-solver",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    cpu=4.0,  # 4 CPUs
    memory=8192,  # 8GB RAM
    timeout=7200,  # 2 hours max
    gpu="T4",  # Optional: add GPU support
)
def train(
    config_id: str,
    max_env_steps: int = None,
    wandb_mode: str = "online",
    rom_game_id: str = None,
):
    """Run training on Modal.

    Args:
        config_id: Environment:variant config (e.g., 'Retro-SuperMarioBros-Nes:ppo')
        max_env_steps: Max environment steps (optional override)
        wandb_mode: W&B mode (online, offline, disabled)
        rom_game_id: Retro game ID for ROM import (e.g., 'SuperMarioBros-Nes')
    """
    import subprocess

    # Package already installed in image from pyproject.toml
    os.chdir("/tmp/gymnasium-solver")
    print(f"Working directory: {os.getcwd()}")

    # If this is a Retro environment, import the ROM first
    if rom_game_id:
        print(f"Importing Retro ROM for game: {rom_game_id}")
        rom_path = f"/tmp/retro-roms/{rom_game_id}"
        if not Path(rom_path).exists():
            raise RuntimeError(f"ROM directory not found at {rom_path}")

        # Import the ROM using retro.import
        import_cmd = ["python", "-m", "retro.import", rom_path]
        print(f"Running: {' '.join(import_cmd)}")
        result = subprocess.run(import_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"ROM import failed with return code {result.returncode}")

        print("ROM imported successfully")

    # Set environment variables
    os.environ["VIBES_QUIET"] = "1"
    os.environ["VIBES_DISABLE_SESSION_LOGS"] = "1"
    os.environ["WANDB_MODE"] = wandb_mode

    # Build command
    cmd = ["python", "train.py", config_id]
    if max_env_steps:
        cmd.extend(["--max-env-steps", str(max_env_steps)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")

    print("Training completed successfully")


def _is_retro_env(config_id: str) -> bool:
    """Check if config_id is for a Retro environment."""
    # Extract env_id from config_id (format: env_id:variant or project_id:variant)
    env_part = config_id.split(":")[0]
    return env_part.lower().startswith("retro-") or env_part.lower().startswith("retro/")


def _extract_game_id_from_config(config_id: str) -> str:
    """Extract game ID from Retro config_id.

    Examples:
        'Retro-SuperMarioBros-Nes:ppo' -> 'SuperMarioBros-Nes'
        'Retro/SuperMarioBros-Nes:ppo' -> 'SuperMarioBros-Nes'
    """
    env_part = config_id.split(":")[0]
    # Remove 'Retro-' or 'Retro/' prefix
    if env_part.lower().startswith("retro-"):
        return env_part[6:]  # len('Retro-') = 6
    elif env_part.lower().startswith("retro/"):
        return env_part[6:]  # len('Retro/') = 6
    return env_part


@app.local_entrypoint()
def main(
    config_id: str,
    max_env_steps: int = None,
    wandb_mode: str = "online",
):
    """Launch training on Modal.

    Args:
        config_id: Environment:variant config (e.g., 'Retro-SuperMarioBros-Nes:ppo')
        max_env_steps: Max environment steps (optional override)
        wandb_mode: W&B mode (online, offline, disabled)
    """
    print(f"Launching training: {config_id}")
    if max_env_steps:
        print(f"Max env steps: {max_env_steps}")
    print(f"W&B mode: {wandb_mode}")

    # Check if this is a Retro environment
    rom_mount = None
    rom_game_id = None

    if _is_retro_env(config_id):
        print("Detected Retro environment, checking for ROM...")

        # Import the ROM mapping function
        sys.path.insert(0, str(Path(__file__).parent))
        from list_retro_roms import get_retro_games_map

        # Get the game ID
        game_id = _extract_game_id_from_config(config_id)
        print(f"Game ID: {game_id}")

        # Get ROM information
        games_map = get_retro_games_map()
        if game_id not in games_map:
            raise RuntimeError(
                f"Game '{game_id}' not found in stable-retro. "
                f"Available games: {sorted([g for g in games_map.keys() if games_map[g]['rom_exists']])}"
            )

        game_info = games_map[game_id]
        if not game_info["rom_exists"]:
            raise RuntimeError(
                f"ROM for game '{game_id}' not imported. "
                f"Import it with: python -m retro.import /path/to/rom/directory"
            )

        rom_path = game_info["path"]
        print(f"ROM found at: {rom_path}")

        # Create Modal mount for the ROM directory
        rom_mount = modal.mount.Mount.from_local_dir(
            local_path=rom_path,
            remote_path=f"/tmp/retro-roms/{game_id}",
        )
        rom_game_id = game_id
        print(f"ROM will be uploaded to Modal at: /tmp/retro-roms/{game_id}")

    # Launch training with ROM mount if needed
    if rom_mount:
        train.with_options(mounts=[rom_mount]).remote(
            config_id, max_env_steps, wandb_mode, rom_game_id
        )
    else:
        train.remote(config_id, max_env_steps, wandb_mode, None)

    print("\nTraining completed!")
