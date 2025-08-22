import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from utils.env_info_schema import validate_env_info


def find_yaml_files(root: Path) -> List[Path]:
    patterns = ["*.yaml", "*.yml"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    # Deduplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


def validate_file(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return [("root", "YAML must load to a mapping")] 
    return validate_env_info(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate env_info YAML files against schema.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("config/env_info")], help="Paths to directories or files to validate")
    args = parser.parse_args()

    yaml_files: List[Path] = []
    for p in args.paths:
        if p.is_dir():
            yaml_files.extend(find_yaml_files(p))
        elif p.is_file():
            yaml_files.append(p)
        else:
            print(f"warning: path not found: {p}", file=sys.stderr)

    if not yaml_files:
        print("No YAML files found.")
        return 0

    total_errors = 0
    for yf in sorted(yaml_files):
        errors = validate_file(yf)
        if errors:
            total_errors += len(errors)
            print(f"✗ {yf}")
            for path, msg in errors:
                print(f"  - {path}: {msg}")
        else:
            print(f"✓ {yf}")

    if total_errors:
        print(f"Found {total_errors} validation errors across {len(yaml_files)} files.")
        return 1
    print("All files valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
