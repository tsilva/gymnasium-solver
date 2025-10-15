"""Upload Retro ROMs to Modal Volume.

Usage:
    python scripts/upload_rom_to_modal.py SuperMarioBros-Nes
    python scripts/upload_rom_to_modal.py --all
"""

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from list_retro_roms import get_retro_games_map


def upload_rom_to_volume(game_id: str, volume_name: str = "roms"):
    """Upload a single ROM directory to Modal volume.

    Args:
        game_id: Game ID (e.g., 'SuperMarioBros-Nes')
        volume_name: Modal volume name
    """
    # Get ROM path
    games_map = get_retro_games_map()
    if game_id not in games_map:
        available = sorted([g for g in games_map.keys() if games_map[g]['rom_exists']])
        raise RuntimeError(
            f"Game '{game_id}' not found. Available: {available}"
        )

    game_info = games_map[game_id]
    if not game_info["rom_exists"]:
        raise RuntimeError(
            f"ROM for '{game_id}' not imported locally. "
            f"Import with: python -m retro.import /path/to/rom/directory"
        )

    rom_path = Path(game_info["path"])
    if not rom_path.exists():
        raise RuntimeError(f"ROM path not found: {rom_path}")

    print(f"Uploading ROM: {game_id}")
    print(f"Local path: {rom_path}")

    # Get or create volume
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)

    # Upload all files in ROM directory
    with vol.batch_upload() as batch:
        for file_path in rom_path.rglob("*"):
            if file_path.is_file():
                # Preserve directory structure under /retro-roms/<game_id>/
                relative_path = file_path.relative_to(rom_path)
                remote_path = f"/retro-roms/{game_id}/{relative_path}"
                print(f"  Uploading: {relative_path} -> {remote_path}")
                batch.put_file(str(file_path), remote_path)

    print(f"✓ Successfully uploaded {game_id} to volume '{volume_name}'")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    game_id = sys.argv[1]
    volume_name = sys.argv[2] if len(sys.argv) > 2 else "roms"

    if game_id == "--all":
        # Upload all ROMs
        games_map = get_retro_games_map()
        available_games = [g for g in games_map.keys() if games_map[g]['rom_exists']]
        print(f"Uploading {len(available_games)} ROMs to volume '{volume_name}'...")
        for game in available_games:
            upload_rom_to_volume(game, volume_name)
        print(f"\n✓ All ROMs uploaded successfully!")
    else:
        # Upload single ROM
        upload_rom_to_volume(game_id, volume_name)


if __name__ == "__main__":
    main()
