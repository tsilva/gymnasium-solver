# Static analysis sweep

## Goal
Eliminate latent defects by running lightweight static checks and correcting flagged issues across core packages.

## Steps
1. Ensure the code imports cleanly: run `uv run python -m compileall agents utils gym_wrappers trainer_callbacks` (fallback to `python -m compileall ...` if `uv` is unavailable) and address any syntax errors.
2. Execute available linters (e.g., `uv run ruff check .` or `uv run pyflakes .`); when a tool is missing, note it in the summary and perform a manual scan for the same issue class (unused imports, unreachable code, wildcard imports).
3. Grep for red-flag patterns using `rg` (e.g., `except Exception`, `TODO`, `FIXME`, `print(` in library code, `pdb.set_trace`, lingering debug flags) and triage each hit.
4. For each legitimate finding, implement the smallest fix that respects existing styles—avoid sweeping refactors or formatting churn.
5. Re-run the relevant static checks to confirm they are clean.
6. When fixes touch behavior, execute `pytest -q` to confirm runtime correctness.
7. Summarize the tools run, key fixes, and any remaining follow-ups in the task report.

## Notes
- Keep fixes scoped; if a warning uncovers a large design problem, capture it in `TODO.md` or a new task rather than rewriting the module here.
- Do not introduce new dependencies solely for linting without prior approval; prefer optional tooling available via `uv` extras.
- Capture the tool versions (e.g., `ruff --version`) in the summary when discrepancies could stem from local tool drift.
