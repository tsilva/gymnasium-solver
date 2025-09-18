# Audit encapsulation opportunities

## Goal
Inspect the codebase for places where shared behavior can be encapsulated to reduce duplication and hide incidental complexity, while preserving clear interfaces and existing abstractions. Do not change code as part of this task. Capture each opportunity as a `REFACTOR:` item in `TODO.md` with exact file paths and symbols.

## Steps
1. Refresh context: review `README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, `AGENTS.md`, and recent notes (`TODO.md`) to understand current boundaries, extension points, and utilities.
2. Build a candidate list of weak encapsulation:
   - Repeated pre/post-processing across modules that could live behind a helper or adapter.
   - Long parameter lists and plumbing of the same arguments through multiple layers (often a sign a helper/object should own them).
   - Direct access to internals across modules (e.g., `obj._private`), or leaky abstractions where callers must know implementation details.
   - Lifecycle/code that mixes concerns (IO/config parsing tangled with core logic) that should be wrapped behind a simple interface.
   - Similar environment/collector/model construction patterns that could share a common builder.
3. Generate structural hints:
   - `uv run ruff check --select PLR0912,PLR0915,PLR0913` (fallback: `python -m ruff ...`) to flag long/branchy functions and many-argument call sites.
   - Scan for pass-through parameters (same arg name appearing in many adjacent function signatures) and repeated kwargs expansion.
4. Validate each candidate by tracing data flow:
   - Confirm encapsulation would reduce call-site complexity, repetition, or risk of drift.
   - Identify the natural home for the behavior (e.g., `utils/`, `trainer_callbacks/`, method on `BaseAgent` or a wrapper).
   - Note required interfaces and any subtle invariants the helper must own.
5. Prioritize opportunities by impact vs. effort; outline minimal, incremental extractions that preserve public APIs.
6. Add `REFACTOR:` entries to `TODO.md` for each opportunity with:
   - Exact file paths and symbols (e.g., `trainer_callbacks/console_summary.py:print_summary`).
   - Proposed interface/home module and acceptance criteria (unchanged behavior, tests pass).
   - Risks/assumptions and a rough size estimate (S/M/L). Do not refactor code as part of this audit.

## Search tips
- Find potential private/internal access: `rg -n "\._[A-Za-z]"`.
- Spot wide function signatures: `rg -n "def [^(]+\([^)]{60,}\)"` (tune threshold).
- Locate repeated setup/teardown patterns across `utils/`, `agents/`, `trainer_callbacks/`, and CLI scripts.

## Notes
- No code changes in this task—produce `REFACTOR:` items only.
- Align encapsulation with existing module boundaries and registries; prefer improving current homes over adding new single-use helpers.
- Call out risks like tighter coupling, hidden configuration, or testing gaps; suggest how the encapsulated API would mitigate them.
- When runtime behavior relies on reflection/registries, document how a shared helper preserves current semantics.
- If tooling like `uv` or `ruff` is unavailable, note the fallback command or manual review performed.
