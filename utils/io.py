"""UTF-8 JSON/YAML file IO helpers.

Provides small, reusable functions to read/write JSON and YAML files
with explicit UTF-8 encoding across the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import json
import yaml

PathLike = Union[str, Path]


def _to_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def read_json(path: PathLike, *, encoding: str = "utf-8") -> Any:
    p = _to_path(path)
    with p.open("r", encoding=encoding) as f: data = json.load(f)
    return data


def write_json(
    path: PathLike,
    data: Any,
    *,
    indent: int = 2,
    encoding: str = "utf-8",
    ensure_dirs: bool = True,
    **kwargs,
) -> None:
    p = _to_path(path)
    if ensure_dirs: p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding=encoding) as f: json.dump(data, f, indent=indent, **kwargs)


def read_yaml(path: PathLike, *, encoding: str = "utf-8") -> Any:
    p = _to_path(path)
    with p.open("r", encoding=encoding) as f: data = yaml.safe_load(f)
    return data


def write_yaml(
    path: PathLike,
    data: Any,
    *,
    encoding: str = "utf-8",
    ensure_dirs: bool = True,
    **kwargs,
) -> None:
    p = _to_path(path)
    if ensure_dirs: p.parent.mkdir(parents=True, exist_ok=True)
    dumper = yaml.safe_dump if kwargs.get("safe", True) else yaml.dump
    text = dumper(data, **kwargs)
    p.write_text(text, encoding=encoding)

