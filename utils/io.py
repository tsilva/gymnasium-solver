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


def read_text_utf8(path: PathLike) -> str:
    p = _to_path(path)
    return p.read_text(encoding="utf-8")


def write_text_utf8(path: PathLike, text: str) -> None:
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def read_json(path: PathLike) -> Any:
    p = _to_path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    path: PathLike,
    data: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
    sort_keys: bool | None = None,
    default: Any | None = None,
) -> None:
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    if indent is not None:
        kw["indent"] = indent
    if sort_keys is not None:
        kw["sort_keys"] = sort_keys
    if default is not None:
        kw["default"] = default
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, **kw)


def read_yaml(path: PathLike) -> Any:
    p = _to_path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(
    path: PathLike,
    data: Any,
    *,
    safe: bool = True,
    sort_keys: bool = False,
    indent: int = 2,
) -> None:
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dumper = yaml.safe_dump if safe else yaml.dump
    text = dumper(data, sort_keys=sort_keys, indent=indent, allow_unicode=True)
    p.write_text(text, encoding="utf-8")

