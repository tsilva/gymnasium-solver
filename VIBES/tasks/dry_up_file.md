# DRY up a single file

## Goal
Reduce duplication within a single module by extracting shared behavior into clear helpers while keeping the file cohesive and readable.

## Steps
1. Prep context: reread `README.md`, `VIBES/INTERNALS.md`, and any notes in `AGENTS.md` / `TODO.md` that mention the target module. Run `git status` to confirm a clean baseline and skim existing tests for the file.
2. Map the duplication: open the file, mark repeated blocks, and use `rg --context 3 "<key phrase>" <file>` (or structural search in the IDE) to verify every occurrence you plan to unify. Capture how each variant differs (inputs, conditionals, side effects).
3. Design the extraction: look for existing helpers or utilities elsewhere in the codebase that already provide the shared behavior; prefer reusing or lightly extending them when they fit. When no suitable helper exists, decide on the minimal new helper (function, small dataclass, or inline utility) that captures the shared behavior without over-generalizing. Write down the interface (args, expected behavior, returned data) and note any invariants or exceptions it should enforce.
4. Implement carefully: introduce the helper, refactor call sites one at a time, and keep related logic adjacent so future readers can trace the flow. Preserve existing formatting, logging, and error handling; only touch lines needed for the refactor. Add short docstrings or comments when intent is not obvious.
5. Validate behavior: update or add focused tests covering the helper and each refactored path. Run the relevant suite (default: `pytest -q`; or a narrower target like `pytest <file>::<test>`). Fix any regressions before proceeding.
6. Wrap up: ensure docs or configs that reference the old structure are still accurate, update `TODO.md` if follow-up items remain, and summarize the change (including the root-cause for the duplication) in your PR or handoff notes.

## Notes
- Prefer keeping helpers local to the module unless multiple modules already need them; avoid premature moves into `utils/`.
- Resist creating configuration flags or deep parameter lists just to support edge cases—refactor the call sites instead.
- Maintain initialization order and side effects; confirm that shared helpers execute in the same sequence as before.
- If you cannot safely consolidate a block, document why and leave the duplication in place rather than breaking behavior.
