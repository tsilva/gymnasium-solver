## AGENTS: Workspace Rules and Operating Guide

Rules for autonomous and assisted agents working in this repository. Follow these in order to make safe, minimal, and helpful changes.

### Before You Start
- Read `VIBES/ARCHITECTURE_GUIDE.md`, `VIBES/CODING_PRINCIPLES.md`, and `README.md` end-to-end.
- Task system: playbooks live under `VIBES/tasks/`.
  - `run task: <name>` maps to `VIBES/tasks/<name>.md` (normalize: lowercase, spaces→underscores, strip punctuation).
  - `!TASK: <name>` always refers to a single file in `VIBES/tasks/`; execute every instruction in that file.
- Do not modify `VIBES/tasks/*.md` while executing a `!TASK`. Capture notes separately and update later only when explicitly asked.

### Decision Hierarchy
- Safety first: security, privacy, and data integrity override everything.
- User intent: follow explicit current-session instructions unless unsafe.
- This guide: when intent is ambiguous, follow these rules.
- Codebase conventions: match existing styles and patterns.

### Allowed Actions
- Read: freely read workspace files for context.
- Edit: make minimal, focused edits tied to the request or to fix issues introduced by your edits.
- Create files: when necessary (configs, docs, tests, small helpers); co-locate with similar files.
- Delete: only when explicitly requested or when replacing clearly obsolete generated artifacts.

### Communication
- Be concise; add detail only when needed.
- Use headings and bullets for readability.
- Show code only when essential; use fenced blocks for new code and include exact paths when citing existing files.
- End each turn with a brief edit summary: what changed and why (2–6 bullets).

### Editing & Code Changes
- Minimal diffs: change only what’s necessary; avoid drive-by refactors.
- Preserve formatting: keep indentation style and width; don’t reflow unrelated lines.
- Naming: prefer descriptive, clear identifiers.
- Imports and deps: add needed imports and update configs/dependencies if required.
- Tests: add or update tests when behavior changes or is newly added.
- Docs/config: update `README.md and relevant configs when user-facing behavior or defaults change.

### Root-Cause-First Changes
- Diagnose: trace failures to their true cause by reading adjacent modules and following data flow; don’t guess.
- Plan minimal fix: choose the smallest targeted change that fixes the root cause and note the intended change before editing.
- No symptom patches: avoid hardcoded values, special-case branches, broad try/excepts, or duplicated logic that hides bugs.
- Avoid bloat: don’t introduce knobs/flags or helper layers unless they directly eliminate the root cause.
- Prove it: reproduce the issue and add/adjust a focused test or check; make it pass for the right reason.
- Explain why: include a one-line root-cause statement in your summary and why the fix is the minimal, correct one.

### Documentation
- After tasks, update `VIBES/ARCHITECTURE_GUIDE.md` and `README.md` when relevant.
- Never modify `VIBES/CODING_PRINCIPLES.md` unless explicitly instructed.

### Python/Project Conventions
- Prefer explicit, readable code and early returns.
- Add short docstrings for non-trivial functions; explain “why” when intent isn’t obvious.
- Match existing module layout and patterns (e.g., `utils/`, `agents/`, `gym_wrappers/`).

### Exceptions
- Fail fast: don’t silence unexpected errors; let them surface.
- Never use broad catches (`except Exception:`) or bare `except:`. Don’t use `pass`, sentinel returns, or silent fallbacks that mask bugs.
- Catch only specific, expected cases:
  - Soft dependencies: raise a clear `ImportError` with guidance.
  - Cleanup/finalizers: log context without masking the primary error.
  - Expected control flow: catch narrowly and re-raise with context if unhandled.
- Keep catch scopes minimal and match exact exception types.

### Commands & Environment
- Assume non-interactive shells; use `--yes`/`--non-interactive` flags when available.
- If a command pages output, append `| cat`.
- Run long-lived services in the background.
- Default working directory is the repo root; prefer absolute paths when practical.
- Avoid destructive operations or seek explicit confirmation first.

### Validation & Checks
- Explore with semantic search; use exact search for precise symbols.
- After code edits, ensure the project still builds and tests pass when feasible:
  - Default: `pytest -q` (or `uv run pytest -q` when using `uv`).
  - Fix introduced linter/test failures when reasonably scoped.

### Data, Privacy, and Safety
- Don’t exfiltrate secrets, tokens, environment variables, or private data.
- Don’t access external networks or services unless explicitly requested.
- Avoid commands that alter the system outside this repo unless explicitly requested and safe.

### When Uncertain
- If blocked by missing info, ask one focused question and propose a sensible default.
- When multiple approaches exist, briefly list trade-offs and choose the least risky/complex.
- Prefer asking for clarity over shipping workarounds that paper over issues.

### Repo-Specific Guidance
- Respect `tests/` layout and add tests near related modules.
- Keep env/wrapper logic consistent with `gym_wrappers/` patterns and registries.
- Align new agents or training logic with patterns in `agents/` and `utils/`.

### Prohibited
- Bulk refactors or unrelated formatting changes.
- New runtime dependencies without necessity or explanation.
- Deleting or overwriting user work without explicit instruction.
- Symptom-masking hacks (hardcoded values, broad exception swallowing, duplicate branches).

### Change Management
- Group related changes into coherent edits with brief rationales.
- Prefer small, reviewable steps over large rewrites.

### Quick Checklists
**Before editing**
- Clarify goal and constraints; scan related files for context.
- Identify and state the suspected root cause.
- Plan a minimal intervention that addresses that root cause.

**After editing**
- Validate imports, types, and obvious edge cases.
- Run tests/build if feasible; ensure no new failures.
- Summarize changes and their impact concisely.
- Add or update a regression test that proves the fix when reasonable.
