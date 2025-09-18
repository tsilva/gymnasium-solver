"""Filesystem helpers (paths, symlinks, portability).

Small, reusable helpers for working with files and symlinks in a portable way.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def update_symlink(link_path: Path, target_path: Path) -> None:
    """Create or update a symlink at ``link_path`` pointing to ``target_path``.

    Uses a relative target path from the link's parent directory for portability.
    Ensures parent directories exist. Falls back to copying on platforms that
    do not support symlinks or when permissions are insufficient.
    """
    link_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing link/file if present
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
    except FileNotFoundError:
        pass

    # Compute relative target from link directory for robustness
    rel_target = os.path.relpath(str(target_path), start=str(link_path.parent))

    try:
        link_path.symlink_to(rel_target)
    except (NotImplementedError, OSError, PermissionError):
        # Symlinks not supported or blocked; copy file as a fallback.
        shutil.copy2(target_path, link_path)

