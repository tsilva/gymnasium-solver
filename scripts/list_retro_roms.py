#!/usr/bin/env python3
"""List imported ROMs for stable-retro and their locations."""

import sys
from pathlib import Path


def get_retro_games_map() -> dict[str, dict]:
    """
    Get a mapping of retro game IDs to their information.

    Returns:
        dict: Maps game_id -> {
            'path': Path,
            'rom_path': Path | None,
            'rom_exists': bool,
            'metadata_files': list[str],
            'scenarios': list[str],
            'states': list[str],
        }
    """
    try:
        import retro
    except ImportError:
        return {}

    games_map = {}
    games = sorted(retro.data.list_games())

    for game in games:
        try:
            rom_path = Path(retro.data.get_romfile_path(game))
            game_dir = rom_path.parent
            rom_exists = rom_path.exists()

            # Get metadata files
            metadata_files = []
            if game_dir.exists():
                metadata_files = [
                    f.name for f in sorted(game_dir.iterdir()) if f.is_file() and f != rom_path
                ]

            # Get scenarios and states
            scenarios = list(retro.data.list_scenarios(game))
            states = list(retro.data.list_states(game))

            games_map[game] = {
                "path": game_dir,
                "rom_path": rom_path if rom_exists else None,
                "rom_exists": rom_exists,
                "metadata_files": metadata_files,
                "scenarios": scenarios,
                "states": states,
            }

        except Exception:
            # Game has no ROM file
            games_map[game] = {
                "path": None,
                "rom_path": None,
                "rom_exists": False,
                "metadata_files": [],
                "scenarios": [],
                "states": [],
            }

    return games_map


def main():
    try:
        import retro
    except ImportError:
        print("ERROR: stable-retro is not installed.")
        print("Install it with: uv pip install -e '.[retro]'")
        sys.exit(1)

    print("Stable-Retro ROM Locations")
    print("=" * 80)
    print()

    # Get the data path where ROMs are stored
    data_path = Path(retro.data.DATA_PATH)
    print(f"Data directory: {data_path}")
    print()

    # Get games map
    games_map = get_retro_games_map()

    if not games_map:
        print("No games/ROMs found.")
        print()
        print("To import ROMs, use:")
        print("  python -m retro.import /path/to/rom/directory")
        return

    # Count games with ROMs
    games_with_roms = sum(1 for info in games_map.values() if info["rom_exists"])

    print(f"Found {len(games_map)} game(s) ({games_with_roms} with ROMs imported):")
    print()

    # Display games from the map
    for game_id, info in games_map.items():
        if not info["rom_exists"]:
            # Skip games without ROMs in main output (too verbose)
            continue

        print(f"  {game_id}")
        print(f"    Location: {info['path']}")

        if info["rom_path"]:
            rom_size = info["rom_path"].stat().st_size
            rom_size_kb = rom_size / 1024
            print(f"    ROM: {info['rom_path'].name} ({rom_size_kb:.1f} KB)")

        if info["metadata_files"]:
            print(f"    Metadata: {', '.join(info['metadata_files'])}")

        if info["scenarios"]:
            print(f"    Scenarios: {', '.join(info['scenarios'])}")

        if info["states"]:
            print(f"    States: {', '.join(info['states'])}")

        print()

    # Show count of games without ROMs
    games_without_roms = len(games_map) - games_with_roms
    if games_without_roms > 0:
        print(f"  (+ {games_without_roms} additional game(s) without ROMs imported)")
        print()

    print("=" * 80)
    print()
    print("To import ROMs:")
    print("  python -m retro.import /path/to/rom/directory")
    print()
    print("To import a specific ROM:")
    print("  python -m retro.import --game GameName /path/to/rom.md")


if __name__ == "__main__":
    main()
