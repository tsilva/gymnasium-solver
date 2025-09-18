# Audit DRYness (duplication)

## Goal
Identify copy-paste and near-duplicate logic, constants, and workflows, and propose minimal consolidation paths. Do not change code as part of this task. Capture each opportunity as a `REFACTOR:` item in `TODO.md` with exact file paths and symbols.

## Steps
1. Refresh context: review `README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, `AGENTS.md`, and recent notes (`TODO.md`) to understand current utilities, boundaries, and known debt.
2. Build a candidate list of duplicate or near-duplicate code:
   - Use `rg` to search for repeated identifiers/phrases (e.g., `rollout`, `checkpoint`, `env_wrappers`, repeated error messages, magic numbers).
   - Spot lookalike functions across files (e.g., `rg -n "^\s*def " | cut -d: -f3 | sort | uniq -d` then inspect call sites).
   - Generate structural hints with `uv run ruff check --select PLR0912,PLR0915,PLR0913` (fallback: `python -m ruff ...`) to flag long/branchy functions where duplication often hides.
   - Compare helpers in `utils/`, `agents/`, `trainer_callbacks/`, `gym_wrappers/`, and CLI scripts for similar pre/post-processing flows.
3. Validate each candidate:
   - Diff the snippets to confirm true duplication vs. context-specific variations.
   - Trace data flow to see whether differences are incidental and could be parameterized.
   - Note any shared constants/hard-coded values that should become a single source of truth.
4. Propose consolidation paths (do not refactor in this audit):
   - Extract/centralize helpers into existing modules (prefer established homes over new single-use files).
   - Parameterize small differences via function args or config rather than branches.
   - Deduplicate constants into `utils/` or appropriate module-level constants.
5. Prioritize opportunities by impact vs. effort, capturing proposed target locations/APIs and risks.
6. Add `REFACTOR:` entries to `TODO.md` for each confirmed duplication with:
   - Exact file paths and symbols for both the sources and the proposed consolidated home.
   - Minimal extraction/parameterization plan and acceptance criteria (e.g., identical behavior, tests unchanged).

## Search tips
- Use `rg -n "pattern" --glob '!runs' --glob '!*.ipynb'` to avoid noise.
- Grep for repeated literals or messages to uncover copy-paste blocks (e.g., `rg -n "epoch=<"`).
- List same-named functions across modules to spot duplication in parallel implementations.

## Notes
- No code changes in this task—produce `REFACTOR:` items only.
- Favor consolidation that aligns with existing module boundaries; avoid single-use abstractions.
- Prefer simple, composable utilities over deep helpers that hide behavior.
- If tools like `uv` or `ruff` are unavailable, note the fallback or manual review performed.
