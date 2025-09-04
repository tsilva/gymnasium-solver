#!/usr/bin/env python3
"""
Shim for the Inspector app.

- When executed as a script: launches the Gradio inspector UI.
- When imported as a module (e.g., by libraries importing Python's stdlib `inspect`):
  forwards all attributes to the real stdlib `inspect` implementation to avoid breakage.
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
import os
import sysconfig
from types import ModuleType


def _load_stdlib_inspect() -> ModuleType:
    """Load and return the stdlib `inspect` module by file path.

    Using a file-backed spec avoids importing this shim recursively.
    Raises ImportError on failure.
    """
    stdlib_dir = sysconfig.get_path("stdlib") or sysconfig.get_paths().get("stdlib")
    if not stdlib_dir:
        raise ImportError("Cannot locate stdlib directory to load real 'inspect'.")

    real_path = os.path.join(stdlib_dir, "inspect.py")
    spec = spec_from_file_location("inspect", real_path)
    if not spec or not spec.loader:
        raise ImportError("Failed to build spec for stdlib 'inspect'.")

    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _forward_to_stdlib_inspect() -> None:
    """Populate this module's globals with stdlib inspect attributes.

    Preserves this shim's dunder attributes to keep import semantics stable.
    """
    real_mod = _load_stdlib_inspect()

    preserve = {
        "__name__",
        "__file__",
        "__package__",
        "__spec__",
        "__loader__",
        "__cached__",
        "__doc__",
    }
    for k, v in real_mod.__dict__.items():
        if k not in preserve:
            globals()[k] = v


if __name__ != "__main__":
    # Forward attribute access to the real stdlib inspect when imported.
    try:
        _forward_to_stdlib_inspect()
    except Exception as exc:  # Be explicit and fail loudly for easier debugging
        raise ImportError(f"inspect shim failed to load stdlib inspect: {exc}") from exc
else:
    # Run the app launcher when executed directly
    try:
        from inspector_app import main as _main
    except Exception as e:
        print(f"Failed to start inspector app: {e}")
        raise SystemExit(1)
    _main()
