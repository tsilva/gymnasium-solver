# Static analysis sweep

## Goal
Identify latent defects using lightweight static checks and document precise remediation steps. Do not change code as part of this task. Capture each confirmed issue as an actionable `REFACTOR:`/`CLEANUP:` item in `TODO.md` with exact file paths and symbols.

## Steps
1. Ensure the code imports cleanly: run `uv run python -m compileall agents utils gym_wrappers trainer_callbacks` (fallback to `python -m compileall ...` if `uv` is unavailable) and address any syntax errors.
2. Execute available linters (e.g., `uv run ruff check .` or `uv run pyflakes .`); when a tool is missing, note it in the summary and perform a manual scan for the same issue class (unused imports, unreachable code, wildcard imports). Do not apply fixes.
3. Grep for red-flag patterns using `rg` (e.g., `except Exception`, `TODO`, `FIXME`, `print(` in library code, `pdb.set_trace`, lingering debug flags) and triage each hit.
4. For each legitimate finding, add a `REFACTOR:`/`CLEANUP:` entry to `TODO.md` that includes:
   - Exact file paths and symbols (e.g., `utils/io.py:write_json`).
   - The minimal corrective action and rationale.
   - Acceptance criteria (clean linter output; `pytest -q` when behavior may be affected).
5. Do not implement changes during this audit; ensure tasks are self-contained and minimal.
6. Summarize the tools run, key findings, and any remaining follow-ups in the task report or `TODO.md` links.

## Notes
- No code changes in this task—produce `REFACTOR:`/`CLEANUP:` items only.
- Keep proposed fixes scoped; if a warning uncovers a design problem, capture it as a follow-up task rather than broad refactors.
- Do not introduce new dependencies solely for linting; prefer optional tooling available via `uv` extras in follow-ups.
- Capture tool versions (e.g., `ruff --version`) in notes when helpful for reproducing results.
