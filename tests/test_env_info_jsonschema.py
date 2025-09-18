from pathlib import Path
from typing import List

import pytest
import yaml

try:
    import json

    from jsonschema import Draft202012Validator
    JSONSCHEMA_AVAILABLE = True
except Exception:  # pragma: no cover
    JSONSCHEMA_AVAILABLE = False


@pytest.mark.skipif(not JSONSCHEMA_AVAILABLE, reason="jsonschema not installed")
def test_env_info_files_conform_to_jsonschema():
    schema_path = Path("schemas/env_info.schema.json")
    assert schema_path.exists(), "Schema file missing: schemas/env_info.schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema)

    root = Path("config/environments")
    assert root.exists(), "config/environments directory not found"

    def find_yaml_files(dir_path: Path) -> List[Path]:
        files: List[Path] = []
        # Only consider spec files
        for pattern in ("*.spec.yaml", "*.spec.yml"):
            files.extend(dir_path.rglob(pattern))
        # Deduplicate
        seen = set()
        unique: List[Path] = []
        for f in files:
            if f not in seen:
                unique.append(f)
                seen.add(f)
        return unique

    failures: List[str] = []
    for yf in sorted(find_yaml_files(root)):
        with yf.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"env_info YAML must be a mapping: {yf}"
        errs = list(validator.iter_errors(data))
        if errs:
            formatted = "\n".join([f"  - {'.'.join([str(x) for x in e.absolute_path]) or 'root'}: {e.message}" for e in errs])
            failures.append(f"{yf}:\n{formatted}")

    assert not failures, "Some env_info files fail JSON Schema validation:\n" + "\n".join(failures)
