from pathlib import Path
from typing import List, Tuple

import yaml

from utils.env_info_schema import validate_env_info


def _find_yaml_files(root: Path) -> List[Path]:
    files: List[Path] = []
    # Only consider spec files
    for pattern in ("*.spec.yaml", "*.spec.yml"):
        files.extend(root.rglob(pattern))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique


def _validate_file(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"env_info YAML must be a mapping: {path}"
    return validate_env_info(data)


def test_env_info_files_are_valid():
    root = Path("config/environments")
    assert root.exists(), "config/environments directory not found"

    failures: List[str] = []
    for yf in sorted(_find_yaml_files(root)):
        errors = _validate_file(yf)
        if errors:
            failures.append("\n".join([f"  - {path}: {msg}" for path, msg in errors]))
    assert not failures, "Some env_info files are invalid:\n" + "\n".join(failures)
