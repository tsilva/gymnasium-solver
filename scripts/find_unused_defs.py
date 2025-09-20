#!/usr/bin/env python3
"""Heuristic finder for potentially unused Python functions/methods in this repo."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def iter_py_files(base: str) -> Iterable[str]:
    skip_dirs = {".git", "__pycache__", "runs", "wandb", "stable-retro"}
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(root, f)


@dataclass
class DefSite:
    path: str
    lineno: int
    qualname: str  # e.g., module:function or module:Class.method
    name: str      # simple name
    kind: str      # "function" | "method"


def parse_defs(py_path: str) -> List[DefSite]:
    rel_path = os.path.relpath(py_path, REPO_ROOT)
    with open(py_path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=rel_path)
    except SyntaxError:
        return []

    defs: List[DefSite] = []
    module = rel_path.replace(os.sep, "/")

    class ClassVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: List[str] = []
            self.func_stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            # methods
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = b.name
                    if name.startswith("__") and name.endswith("__"):
                        continue
                    defs.append(
                        DefSite(
                            path=rel_path,
                            lineno=b.lineno,
                            qualname=f"{module}:{node.name}.{name}",
                            name=name,
                            kind="method",
                        )
                    )
            # nested classes/functions handled by generic traversal
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Only consider top-level (module) functions: not inside any class or function
            if not self.class_stack and not self.func_stack:
                name = node.name
                if not (name.startswith("__") and name.endswith("__")):
                    defs.append(
                        DefSite(
                            path=rel_path,
                            lineno=node.lineno,
                            qualname=f"{module}:{name}",
                            name=name,
                            kind="function",
                        )
                    )
            # Track nested function scope while visiting children
            self.func_stack.append(node.name)
            try:
                self.generic_visit(node)
            finally:
                self.func_stack.pop()

    ClassVisitor().visit(tree)
    return defs


def load_text_map(paths: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        rel = os.path.relpath(p, REPO_ROOT)
        try:
            with open(p, "r", encoding="utf-8") as f:
                out[rel] = f.read()
        except Exception:
            continue
    return out


def is_used(defn: DefSite, text_map: Dict[str, str]) -> bool:
    # Consider any reference in any file as usage, excluding the exact def site line
    # Heuristics differ for methods and functions
    name = re.escape(defn.name)
    if defn.kind == "function":
        patterns = [
            rf"\b{name}\s*\(",   # direct call
            rf"@\s*{name}\b",     # decorator usage
            rf"^\s*from\s+.+\s+import\s+.*\b{name}\b",  # direct import
            rf"^\s*import\s+.*\b{name}\b",               # module import alias (best-effort)
        ]
    else:
        # method: look for `.name` or `Class.name` or direct call like `.name(`
        patterns = [rf"\.{name}\b", rf"\b{name}\s*\("]

    def_line_key = (defn.path, defn.lineno)
    for rel_path, text in text_map.items():
        # Scan per-line to allow skipping the def line
        for i, line in enumerate(text.splitlines(), start=1):
            if (rel_path, i) == def_line_key:
                continue
            for pat in patterns:
                if re.search(pat, line):
                    return True
    return False


def main() -> None:
    files = list(iter_py_files(REPO_ROOT))
    # Include tests in references to avoid false positives
    text_map = load_text_map(files)

    # Parse defs excluding tests/__init__.py
    def_sites: List[DefSite] = []
    for f in files:
        rel = os.path.relpath(f, REPO_ROOT)
        # Skip test function definitions themselves (but still count references in tests)
        if rel.startswith("tests/"):
            continue
        def_sites.extend(parse_defs(f))

    unused: List[DefSite] = []
    for d in def_sites:
        if is_used(d, text_map):
            continue
        unused.append(d)

    if not unused:
        print("No unused defs found by heuristic.")
        return

    print("Potentially unused definitions (review before removal):")
    for d in sorted(unused, key=lambda x: (x.path, x.lineno)):
        print(f"- {d.path}:{d.lineno}  {d.kind}  {d.qualname}")


if __name__ == "__main__":
    main()
