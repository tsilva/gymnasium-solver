"""Modal AI runner for single training runs.

Usage:
    # Non-Retro environments
    modal run scripts/modal_train.py --config-id CartPole-v1:ppo --max-env-steps 50000

    # Retro environments (requires ROMs uploaded to Modal volume)
    python scripts/upload_rom_to_modal.py SuperMarioBros-Nes
    modal run scripts/modal_train.py --config-id Retro-SuperMarioBros-Nes:ppo
"""

import os

import modal

# Define Modal app
app = modal.App("gymnasium-solver-train")

# Volume for Retro ROMs
roms_volume = modal.Volume.from_name("roms", create_if_missing=True)

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
    volumes={"/roms": roms_volume},
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
        config_id: Environment:variant config (e.g., 'CartPole-v1:ppo')
        max_env_steps: Max environment steps (optional override)
        wandb_mode: W&B mode (online, offline, disabled)
        rom_game_id: Retro game ID for ROM import (e.g., 'SuperMarioBros-Nes')
    """
    import subprocess
    from pathlib import Path

    # Package already installed in image from pyproject.toml
    os.chdir("/tmp/gymnasium-solver")
    print(f"Working directory: {os.getcwd()}")

    # If this is a Retro environment, import the ROM from volume
    if rom_game_id:
        print(f"Importing Retro ROM for game: {rom_game_id}")
        rom_path = Path(f"/roms/retro-roms/{rom_game_id}")

        if not rom_path.exists():
            raise RuntimeError(
                f"ROM directory not found in volume at {rom_path}. "
                f"Upload ROM with: python scripts/upload_rom_to_modal.py {rom_game_id}"
            )

        # Import the ROM using retro.import
        import_cmd = ["python", "-m", "retro.import", str(rom_path)]
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
        config_id: Environment:variant config (e.g., 'CartPole-v1:ppo')
        max_env_steps: Max environment steps (optional override)
        wandb_mode: W&B mode (online, offline, disabled)
    """
    print(f"Launching training: {config_id}")
    if max_env_steps:
        print(f"Max env steps: {max_env_steps}")
    print(f"W&B mode: {wandb_mode}")

    # Check for Retro environments and extract game ID
    rom_game_id = None
    if _is_retro_env(config_id):
        rom_game_id = _extract_game_id_from_config(config_id)
        print(f"Detected Retro environment: {rom_game_id}")
        print(f"ROM will be loaded from volume: /roms/retro-roms/{rom_game_id}")

    # Launch training
    train.remote(config_id, max_env_steps, wandb_mode, rom_game_id)

    print("\nTraining completed!")
