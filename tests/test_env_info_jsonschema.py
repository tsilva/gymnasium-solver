from pathlib import Path
from typing import Dict, List, Tuple

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

    def find_config_files(dir_path: Path) -> List[Path]:
        files: List[Path] = []
        for pattern in ("*.yaml", "*.yml"):
            files.extend(dir_path.rglob(pattern))
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

    def extract_spec_blocks(path: Path) -> List[Tuple[str, Dict]]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return []
        specs: List[Tuple[str, Dict]] = []
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

    failures: List[str] = []
    for cfg_path in sorted(find_config_files(root)):
        for label, spec in extract_spec_blocks(cfg_path):
            errs = list(validator.iter_errors(spec))
            if errs:
                formatted = "\n".join([f"  - {'.'.join([str(x) for x in e.absolute_path]) or 'root'}: {e.message}" for e in errs])
                failures.append(f"{cfg_path}::{label}:\n{formatted}")

    assert not failures, "Some env_info files fail JSON Schema validation:\n" + "\n".join(failures)
