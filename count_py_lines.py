#!/usr/bin/env python3
"""
Tiny utility: count total lines across repo Python files.

Usage:
  python count_py_lines.py             # count under CWD
  python count_py_lines.py <path>      # count under <path>
  python count_py_lines.py -d [path]   # include per-file details

Notes:
- Skips common non-source and artifact dirs to avoid double-counting
  (e.g., venvs, caches, runs/, wandb/, vendored stable-retro/).
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, List, Sequence, Tuple

# Directories to exclude during traversal
EXCLUDE_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    # Run artifacts and logs
    "runs",
    "wandb",
    "logs",
    # Vendored code
    "stable-retro",
}


def iter_py_files(root: str) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        # In-place filter of dirs to walk into
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.endswith(".py"):
                yield os.path.join(dirpath, fname)


def count_lines(paths: Sequence[str]) -> Tuple[int, List[Tuple[str, int]]]:
    total = 0
    per_file: List[Tuple[str, int]] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                n = sum(1 for _ in fh)
        except OSError:
            n = 0
        per_file.append((p, n))
        total += n
    return total, per_file


def main(argv: Sequence[str]) -> int:
    show_details = any(a in {"-d", "--details"} for a in argv)
    args = [a for a in argv if a not in {"-d", "--details"}]

    root = args[0] if args else os.getcwd()
    root = os.path.abspath(root)

    files = list(iter_py_files(root))
    total, per_file = count_lines(files)

    print(f"Root: {root}")
    print(f"Python files: {len(files)}")
    print(f"Total lines: {total}")

    if show_details:
        for path, n in sorted(per_file, key=lambda x: (-x[1], x[0])):
            rel = os.path.relpath(path, root)
            print(f"{n:7d}  {rel}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

