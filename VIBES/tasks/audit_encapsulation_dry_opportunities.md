# Audit encapsulation opportunities for DRY improvements

## Goal
Inspect the codebase for places where shared behavior can be encapsulated to reduce duplication, while preserving clarity and respecting existing abstractions.

## Steps
1. Refresh context: review `README.md`, `VIBES/INTERNALS.md`, `AGENTS.md`, and recent notes (`TODO.md`) to understand current abstractions, extension points, and known tech debt.
2. Build a candidate list of duplicate or near-duplicate logic:
   - Use `rg` to search for repeated identifiers or phrases (e.g., `rollout`, `checkpoint`, `env_wrappers`) across modules.
   - Generate structural hints with `uv run ruff check --select PLR0912,PLR0915,PLR0913` (fallback: `python -m ruff ...`) to flag overly long functions or argument lists that might benefit from helper encapsulation.
   - Compare helper utilities in `utils/`, `agents/`, `trainer_callbacks/`, and CLI scripts for similar pre/post-processing that could share a common API.
3. For each candidate, trace the data flow to confirm the duplication is real rather than contextual (e.g., separate concerns for train vs eval). Document why encapsulation would help (less repeated params, single source of truth, easier testing).
4. Prioritize opportunities by impact and effort, noting required interfaces (e.g., new helper in `utils/`, abstract method on `BaseAgent`, wrapper method).
5. Record findings in the task report or `TODO.md` with clear next steps. Do **not** refactor code as part of this audit.

## Notes
- Favor encapsulations that align with existing module boundaries; avoid introducing single-use helpers.
- Call out risks such as tight coupling, hidden configuration, or testing gaps that encapsulation could address.
- When dynamics rely on reflection/registries, document how a shared helper would preserve runtime behavior.
- If tooling like `uv` or `ruff` is unavailable, note the fallback command or manual review performed.
