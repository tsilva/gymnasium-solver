import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from utils.env_info_schema import validate_env_info

try:
    import json
    from jsonschema import Draft202012Validator
    JSONSCHEMA_AVAILABLE = True
except Exception:  # pragma: no cover
    JSONSCHEMA_AVAILABLE = False


def find_yaml_files(root: Path) -> List[Path]:
    # Validate only spec files
    patterns = ["*.spec.yaml", "*.spec.yml"]
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


def validate_file_python(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return [("root", "YAML must load to a mapping")] 
    return validate_env_info(data)


def validate_file_jsonschema(path: Path, schema_path: Path) -> List[Tuple[str, str]]:
    with schema_path.open("r", encoding="utf-8") as sf:
        schema = json.load(sf)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return [("root", "YAML must load to a mapping")] 
    validator = Draft202012Validator(schema)
    errors = []
    for err in validator.iter_errors(data):
        loc = ".".join([str(x) for x in err.absolute_path]) or "root"
        errors.append((loc, err.message))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate env spec YAML files against schema.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("config/environments")], help="Paths to directories or files to validate")
    parser.add_argument("--jsonschema", action="store_true", help="Validate using JSON Schema in schemas/env_info.schema.json (optional)")
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

    use_jsonschema = bool(args.jsonschema)
    schema_path = Path("schemas/env_info.schema.json")
    if use_jsonschema:
        if not JSONSCHEMA_AVAILABLE:
            print("jsonschema not installed. Install with 'pip install jsonschema' or omit --jsonschema.")
            return 2
        if not schema_path.exists():
            print(f"Schema file not found: {schema_path}")
            return 2

    total_errors = 0
    for yf in sorted(yaml_files):
        if use_jsonschema:
            errors = validate_file_jsonschema(yf, schema_path)
        else:
            errors = validate_file_python(yf)
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
