# Clean up imports

## Goal
Streamline Python imports across the repo so they reflect actual usage, follow project ordering conventions, and avoid redundant or shadowed symbols.

## Steps
1. Generate a baseline report with `uv run ruff check --select F401,F403,F405` (fallback: `python -m ruff check ...`). Note the files that trigger unused or wildcard import warnings.
2. Inspect each flagged file and remove unused imports, replacing wildcard imports with explicit names when the module is small and stable. Keep intentional side-effect imports; document them with a comment instead of deleting them. For local (in-function) imports, hoist them to the module top when it improves clarity and does not impact startup time or introduce import-time side effects; otherwise keep them local to avoid delaying framework boot.
3. Re-run `uv run ruff check --select F401,F403,F405` to confirm the warnings are gone. If new warnings appear, iterate until clean.
4. Normalize import grouping and alphabetical order via `uv run ruff check --select I --fix`. When automatic sorting conflicts with local style (e.g., grouped `torch` imports), perform minimal manual adjustments to honor the existing pattern.
5. After cleanup, execute targeted tests for impacted packages or `pytest -q` when feasible to make sure no runtime-only imports were removed.
6. Summarize the files touched, rationale for any retained unusual imports, and remaining follow-ups (if any).

## Notes
- Hoist vs local imports:
  - Hoist to top-level when the import is lightweight (stdlib or small helpers), used across multiple functions/paths, and has no import-time side effects or registry work. This improves readability and static analysis without hurting boot time.
  - Keep imports local when they are heavy (e.g., deep ML/graphics libs), optional, or trigger import-time initialization. This defers cost until actually needed and avoids slowing framework boot/CLI startup.
- Python caches imports after first load; repeated local imports are cheap after the first call. The main concern is import-time cost and side effects at process start.
- If removing an import breaks a public API (e.g., `__all__`), update the re-export list accordingly or leave the import in place with a brief comment.
- Avoid adding new dependencies; rely on the existing toolchain (`uv`, `ruff`). If `uv` is unavailable, document the fallback command you used.
- For type-only needs, prefer `from __future__ import annotations` and `if TYPE_CHECKING:` blocks to avoid runtime import cost.
- Preserve formatting; do not run full-file formatters unless required for resolving import blocks.
