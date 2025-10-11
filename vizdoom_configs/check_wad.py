#!/usr/bin/env python3
"""Check if doom.wad is available for VizDoom E1M1."""

import os
import sys
from pathlib import Path

def find_doom_wad():
    """Search for doom.wad in common locations."""
    locations = []
    
    # Check VizDoom scenarios directory
    try:
        import vizdoom as vzd
        pkg_dir = Path(vzd.__file__).parent
        scenarios_dir = pkg_dir / "scenarios"
        doom_wad = scenarios_dir / "doom.wad"
        locations.append(("VizDoom scenarios", doom_wad))
    except ImportError:
        print("Warning: vizdoom package not installed")
        return None
    
    # Check VIZDOOM_SCENARIOS_DIR
    env_dir = os.environ.get("VIZDOOM_SCENARIOS_DIR")
    if env_dir:
        doom_wad = Path(env_dir) / "doom.wad"
        locations.append(("VIZDOOM_SCENARIOS_DIR", doom_wad))
    
    # Check current directory
    doom_wad = Path.cwd() / "doom.wad"
    locations.append(("Current directory", doom_wad))
    
    # Check each location
    found_paths = []
    for name, path in locations:
        if path.is_file():
            found_paths.append((name, path))
    
    return found_paths

if __name__ == "__main__":
    print("Searching for doom.wad...")
    print()
    
    found = find_doom_wad()
    
    if found:
        print(f"✓ Found {len(found)} doom.wad file(s):")
        for name, path in found:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  • {name}: {path} ({size_mb:.2f} MB)")
        print()
        print("You can use VizDoom-E1M1-v0:ppo")
        sys.exit(0)
    else:
        print("✗ doom.wad not found in any standard location")
        print()
        print("See vizdoom_configs/README.md for setup instructions")
        sys.exit(1)
