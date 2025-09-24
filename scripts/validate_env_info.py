import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from utils.env_info_schema import validate_env_info
from utils.io import read_json, read_yaml

try:
    from jsonschema import Draft202012Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    JSONSCHEMA_AVAILABLE = False


def find_config_files(root: Path) -> List[Path]:
    patterns = ["*.yaml", "*.yml"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    seen = set()
    ordered: List[Path] = []
    for f in files:
        if f in seen:
            continue
        if f.name.endswith(".spec.yaml") or f.name.endswith(".spec.yml"):
            continue
        if f.name.endswith(".new.yaml") or f.name.endswith(".new.yml"):
            continue
        ordered.append(f)
        seen.add(f)
    return ordered


def _extract_spec_blocks(path: Path) -> Sequence[Tuple[str, Dict[str, Any]]]:
    data = read_yaml(path)
    if not isinstance(data, dict):
        return []

    specs: List[Tuple[str, Dict[str, Any]]] = []
    base_spec = data.get("spec")
    if isinstance(base_spec, dict):
        specs.append(("spec", dict(base_spec)))
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        variant_spec = value.get("spec")
        if isinstance(variant_spec, dict):
            specs.append((f"{key}.spec", dict(variant_spec)))
    return specs


def validate_spec_python(spec: Dict[str, Any]) -> List[Tuple[str, str]]:
    return validate_env_info(spec)


def validate_spec_jsonschema(spec: Dict[str, Any], schema: Dict[str, Any]) -> List[Tuple[str, str]]:
    validator = Draft202012Validator(schema)
    errors = []
    for err in validator.iter_errors(spec):
        loc = ".".join([str(x) for x in err.absolute_path]) or "root"
        errors.append((loc, err.message))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate env spec YAML files against schema.")
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("config/environments")], help="Paths to directories or files to validate")
    parser.add_argument("--jsonschema", action="store_true", help="Validate using JSON Schema in schemas/env_info.schema.json (optional)")
    args = parser.parse_args()

    config_files: List[Path] = []
    for p in args.paths:
        if p.is_dir():
            config_files.extend(find_config_files(p))
        elif p.is_file():
            config_files.append(p)
        else:
            print(f"warning: path not found: {p}", file=sys.stderr)

    if not config_files:
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
    total_specs = 0
    schema_data: Dict[str, Any] = {}
    if use_jsonschema:
        schema_data = read_json(schema_path)

    for cfg_path in sorted(config_files):
        specs = _extract_spec_blocks(cfg_path)
        if not specs:
            continue
        for label, spec in specs:
            total_specs += 1
            if use_jsonschema:
                errors = validate_spec_jsonschema(spec, schema_data)
            else:
                errors = validate_spec_python(spec)
            display_name = f"{cfg_path}::{label}"
            if errors:
                total_errors += len(errors)
                print(f"✗ {display_name}")
                for path, msg in errors:
                    print(f"  - {path}: {msg}")
            else:
                print(f"✓ {display_name}")

    if total_errors:
        print(f"Found {total_errors} validation errors across {total_specs} spec blocks.")
        return 1
    print("All spec blocks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
