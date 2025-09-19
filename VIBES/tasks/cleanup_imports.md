# Clean up imports

## Goal
Streamline Python imports across the repo so they reflect actual usage, follow project ordering conventions, and avoid redundant or shadowed symbols.

## Steps
1. Generate a baseline report with `uv run ruff check --select F401,F403,F405` (fallback: `python -m ruff check ...`). Note the files that trigger unused or wildcard import warnings.
2. Inspect each flagged file and remove unused imports, replacing wildcard imports with explicit names when the module is small and stable. Keep intentional side-effect imports; document them with a comment instead of deleting them. Prefer imports at the module top. Only keep imports local (inside functions) when there is a clear, high-ROI reason (e.g., heavy/slow dependencies on cold paths, optional/soft dependencies behind config or feature flags, or necessary circular-import avoidance where refactoring is non-trivial). When you keep a local import, add a brief comment with the reason.
3. Re-run `uv run ruff check --select F401,F403,F405` to confirm the warnings are gone. If new warnings appear, iterate until clean.
4. Normalize import grouping and alphabetical order via `uv run ruff check --select I --fix`. When automatic sorting conflicts with local style (e.g., grouped `torch` imports), perform minimal manual adjustments to honor the existing pattern.
5. After cleanup, execute targeted tests for impacted packages or `pytest -q` when feasible to make sure no runtime-only imports were removed.
6. Summarize the files touched, rationale for any retained unusual imports, and remaining follow-ups (if any).

## Notes
- Hoist vs local imports:
  - Default: prefer top-level imports for clarity, static analysis, and discoverability.
  - Inline imports only when there is high ROI: heavy/slow dependencies on cold paths, optional/soft dependencies gated by config or feature flags, or unavoidable circular-import constraints where refactoring is non-trivial. If kept local, add a short comment explaining the rationale.
- Python caches imports after first load; repeated local imports are cheap after the first call. The main concern is import-time cost and side effects at process start.
- If removing an import breaks a public API (e.g., `__all__`), update the re-export list accordingly or leave the import in place with a brief comment.
- Avoid adding new dependencies; rely on the existing toolchain (`uv`, `ruff`). If `uv` is unavailable, document the fallback command you used.
- For type-only needs, prefer `from __future__ import annotations` and `if TYPE_CHECKING:` blocks to avoid runtime import cost.
- Preserve formatting; do not run full-file formatters unless required for resolving import blocks.
