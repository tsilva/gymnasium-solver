from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from utils.env_info_schema import validate_env_info


def _find_config_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in ("*.yaml", "*.yml"):
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


def _extract_spec_blocks(path: Path) -> List[Tuple[str, Dict]]:
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


def test_env_info_files_are_valid():
    root = Path("config/environments")
    assert root.exists(), "config/environments directory not found"

    failures: List[str] = []
    for cfg_path in sorted(_find_config_files(root)):
        for label, spec in _extract_spec_blocks(cfg_path):
            errors = validate_env_info(spec)
            if errors:
                prefix = f"{cfg_path}::{label}"
                formatted = "\n".join([f"  - {path}: {msg}" for path, msg in errors])
                failures.append(f"{prefix}\n{formatted}")
    assert not failures, "Some env_info files are invalid:\n" + "\n".join(failures)
