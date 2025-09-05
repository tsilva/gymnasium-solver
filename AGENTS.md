## AGENTS: Workspace Rules and Operating Guide

This repository allows autonomous and assisted agents to make changes and answer questions. This document defines the default rules those agents must follow.

### Start-of-task requirements
- Before starting any task, read `INTERNALS.md` and `README.md` end-to-end to ensure up-to-date context.

### Decision hierarchy
- **Safety first**: Security, privacy, and data integrity rules override all other instructions.
- **User intent next**: Follow explicit user instructions in the current session unless they are unsafe.
- **This file then**: When user intent is ambiguous, follow the rules in this document.
- **Codebase conventions**: Match existing styles and patterns in the repository.

### Scope of allowed actions
- **Reading**: Agents may freely read files in the workspace to gather context.
- **Editing**: Agents may make minimal, focused edits that are directly tied to the user’s request or to fix issues introduced by those edits.
- **Creating files**: Allowed when necessary (e.g., new configs, docs, tests, or small helpers). Prefer co-locating with similar files.
- **Deleting files**: Only when explicitly requested or when replacing generated artifacts that are clearly obsolete.

### Communication standards
- **Be concise by default**; add details only when needed.
- Use `##`/`###` headings, short paragraphs, and bullet lists for readability.
- **Show code** only when essential. Use fenced code blocks for new code and include exact paths when citing existing files.
- **Summarize edits** at the end of a turn: what changed and why, in 2–6 bullets.

### Editing rules
- **Minimal diffs**: Change only what is necessary; avoid drive-by refactors.
- **Preserve formatting**: Keep existing indentation style (tabs vs spaces) and width. Do not reflow or reformat unrelated lines.
- **Naming and clarity**: Prefer descriptive names over abbreviations. Write clear, readable code.
- **Imports and deps**: Add required imports and update configuration/dependency files if needed.
- **Tests**: When behavior changes or is newly added, create or update tests.
- **Config/docs**: Update `README.md`, `EXPERIMENTS.md`, and relevant configs when user-facing behavior or defaults change.

### Root-cause-first changes
- **Diagnose deeply**: Before editing, trace the failing behavior to its true cause. Read adjacent modules, follow data flow, and verify assumptions; do not guess.
- **Plan minimal intervention**: Choose the smallest targeted change that fixes the root cause. Write down the intended change before applying it.
- **No symptom patches**: Do not add hardcoded values, special-case branches, broad try/excepts, or duplicate logic that merely hides the bug.
- **Avoid code bloat**: Do not introduce knobs/flags or helper layers to “make it work” unless they directly eliminate the root cause and are justified.
- **Prove it**: Reproduce the issue, then add/adjust a focused test or reproducible check. The fix should make the test pass for the right reason.
- **Explain why**: In the edit summary, include a one-line root-cause statement and why this is the minimal, correct fix.

### Documentation maintenance
- After completing any task, update `INTERNALS.md` and `README.md` with relevant changes. If no updates are needed, leave them untouched.

### Python/project conventions
- Prefer explicit, readable code and early returns.
- Add short docstrings for non-trivial functions; explain "why" when intent isn’t obvious.
- Match the repository’s existing module layout and patterns (e.g., `utils/`, `agents/`, `gym_wrappers/`).

### Commands and environment
- Assume non-interactive shells. Use `--yes`/`--non-interactive` flags where applicable.
- If a command would use a pager, append `| cat`.
- Long-running services should run in the background.
- Default working directory is the repo root. Prefer absolute paths when possible.
- Before destructive operations, either avoid them or request explicit confirmation.

### Tooling and checks
- Prefer semantic search when exploring the codebase; use exact search for precise symbols.
- After making code edits, ensure the project still builds and tests pass when feasible:
  - Default: `pytest -q` (or `uv run pytest -q` if `uv` is used in the project).
  - Fix introduced linter/test failures before returning, when reasonably scoped.

### Data, privacy, and safety
- Do not exfiltrate secrets, tokens, environment variables, or private data into responses.
- Do not access external networks or services unless explicitly requested.
- Avoid running commands that alter the system outside the repository, unless explicitly requested and safe.

### Decision-making when uncertain
- If blocked by missing information, ask one focused question and propose a sensible default.
- If multiple viable approaches exist, briefly list trade-offs and choose one with the least risk/complexity.
 - Prefer asking for clarity over shipping a workaround that papers over the issue.

### Repository-specific guidance
- Respect the existing testing layout in `tests/` and update or add tests near related modules.
- Keep environment and wrapper logic consistent with `gym_wrappers/` patterns and registries.
- When adding agents or training logic, align with patterns in `agents/` and `utils/`.

### Prohibited actions
- Bulk refactoring or formatting unrelated code.
- Introducing new runtime dependencies without necessity or explanation.
- Deleting or overwriting user work without explicit instruction.
 - Symptom-masking hacks (hardcoded values, broad exception swallowing, duplicate branches) that avoid fixing the root cause.

### Change management
- Group related changes into coherent edits. Include brief rationales in summaries.
- Prefer small, reviewable steps over large rewrites.

### Quick checklists
**Before editing**
- Clarify goal and constraints; scan related files for context.
 - Identify and state the suspected root cause.
 - Plan a minimal intervention that addresses that root cause.

**After editing**
- Validate imports, types, and obvious edge cases.
- Run tests/build if feasible; ensure no new failures.
- Summarize changes and their impact concisely.
 - Add or update a regression test that proves the fix, when reasonable.
