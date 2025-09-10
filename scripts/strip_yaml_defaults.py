#!/usr/bin/env python3
"""
Strip keys in environment YAML files that match Config defaults.
Preserves formatting and comments using ruamel.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from ruamel.yaml import YAML

# Local import
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.config import Config  # noqa: E402

ENV_DIR = Path(__file__).resolve().parents[1] / "config" / "environments"

# Build a dict of Config defaults (only fields with default/default_factory)
DEFAULTS: Dict[str, Any] = {}
for f in Config.__dataclass_fields__.values():
    if f.default is not None and f.default is not f.default_factory:  # type: ignore
        # Note: dataclasses uses MISSING; but utils.config already handles MISSING when assembling defaults
        pass

# The Config class doesn't expose a direct "defaults dict", so create one by instantiating with required fields
# Use innocuous required values; we only need optional fields as default reference.
_required = dict(env_id="DUMMY", algo_id="dummy", n_steps=1, batch_size=1)
_defaults_obj = Config(**_required)
# Collect all attributes and filter out required ones
DEFAULTS = {
    k: getattr(_defaults_obj, k)
    for k in _defaults_obj.__dataclass_fields__.keys()
    if k not in ("env_id", "algo_id", "n_steps", "batch_size")
}

# Keys we should never strip even if equal to defaults, because they define the challenge base
NEVER_STRIP = {
    "project_id",  # Naming anchor for grouping
    "env_id",      # Mandatory
    # Note: algo_id, n_steps, batch_size are typically in child configs; but if present in base, keep
    "algo_id",
}

def normalize_key_value(key: str, value: Any) -> Any:
    """Coerce YAML values to types comparable with Config defaults."""
    if key == "hidden_dims" and isinstance(value, list):
        try:
            return tuple(int(x) for x in value)
        except (TypeError, ValueError):
            return tuple(value)
    return value


def process_mapping(node: Any) -> None:
    # node is a ruamel YAML CommentedMap
    to_delete = []
    for key in list(node.keys()):
        value = node[key]
        if isinstance(value, dict):
            process_mapping(value)
            continue
        # Skip meta keys
        if key in NEVER_STRIP:
            continue
        if key in DEFAULTS:
            norm_v = normalize_key_value(key, value)
            if norm_v == DEFAULTS[key]:
                to_delete.append(key)
    for k in to_delete:
        del node[k]


def main() -> int:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    changed_files = []
    for p in sorted(ENV_DIR.glob("*.yaml")):
        data = yaml.load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        before = yaml.dump(data, sys.stdout.__class__()) if False else None  # no-op
        # Walk all top-level challenge configs
        for k, v in list(data.items()):
            if isinstance(v, dict):
                process_mapping(v)
        new_text_stream = []
        from io import StringIO
        s = StringIO()
        yaml.dump(data, s)
        new_text = s.getvalue()
        old_text = p.read_text(encoding="utf-8")
        if new_text != old_text:
            p.write_text(new_text, encoding="utf-8")
            changed_files.append(p.name)
    print(f"Stripped defaults from {len(changed_files)} files: {', '.join(changed_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
