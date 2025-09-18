# Find and remove dead code

## Goal
Identify unused modules, functions, and branches, then remove them so the codebase stays lean and easier to maintain.

## Steps
1. Review `README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, and recent changes (e.g., `AGENTS.md`, `TODO.md`) to understand active entry points, registered wrappers, and CLI scripts that may reference dynamic hooks.
2. Build a candidate list of suspected dead code:
   - Run `rg --files-with-matches` on telltale markers like `pass  # dead`, `unused`, or comment-out blocks.
   - Search definitions with `rg "def " agents utils gym_wrappers trainer_callbacks scripts` and trace each hit with `rg <symbol>` to confirm no runtime references (include dynamic registries such as `EnvWrapperRegistry`, `agents/__init__.py`, and CLI wiring).
   - When available, use `uv run vulture .` (or `vulture .` if `uv` is unavailable) to surface additional unused symbols; note in the summary if the tool is missing.
3. For every candidate, double-check that it is not loaded via config names, reflection, or tests (e.g., YAML configs, wrapper registries, inspector UI). Document grey areas before deleting.
4. Remove the dead code with minimal diffs—delete the definitions, imports, and related comments/tests. Update import sites, `__all__`, and docs (`README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, task guides) when references are removed.
5. Run `pytest -q` (and targeted smoke scripts such as `python inspector.py --help`) to ensure nothing attempts to import the deleted code.
6. Summarize each removal, the evidence it was unused, and note any follow-up refactors in the task report or `TODO.md`.

## Notes
- Prefer deleting entire obsolete modules over leaving stubs or commented sections; lingering placeholders invite regressions.
- If a function is only used in tests or docs, either restore a real usage or remove the dead usage alongside the function.
- When dynamic import patterns obscure usage, add a quick instrumentation log or targeted test rather than speculating.
- If a removal exposes new abstractions worth extracting, capture that as a separate task—keep this sweep focused on dead code.
