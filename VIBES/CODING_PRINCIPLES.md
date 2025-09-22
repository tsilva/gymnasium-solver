## Coding Principles

This companion to `VIBES/ARCHITECTURE_GUIDE.md` captures the high-level principles we follow when writing or refactoring code. For architecture, data flow, and subsystem ownership, defer to `VIBES/ARCHITECTURE_GUIDE.md`.

### General principles
- **Fail fast, break things**: Never preserve backwards compatibility. Make breaking changes and force users to adapt.
- Respect the config-first workflow: surface behavior through configuration or data models instead of hard-coded branches.
- Compose before reinventing: reach for existing helpers, registries, and extension points so new work integrates cleanly.
- Keep edits scoped: change only what the task requires, and avoid opportunistic refactors that mix concerns.
- **No defensive programming**: Don't add safety checks, validation layers, or compatibility layers. Let the system fail if used incorrectly.

### Python style
- Use type hints consistently, including container generics and optional markers where intent matters.
- Provide short docstrings for non-trivial helpers; reserve inline comments for explaining intent or edge cases.
- Prefer early returns and guard clauses to reduce nesting and keep control flow readable.
- When grouping related fields, reach for small dataclasses or named tuples rather than loosely structured dicts.
- Choose descriptive, unabbreviated identifiers over shorthands (e.g., `current_value` over `curr_val`).
- **Assert aggressively**: Use assertions for all assumptions, preconditions, and invariants. Let the program crash if anything is unexpected.
- **No exception handling**: Never catch exceptions unless absolutely required by the API. Let errors propagate up and crash the program.

### Documentation & configuration
- Centralize file IO through the shared utilities and keep encoding UTF-8 by default.
- Avoid magic values—introduce constants or configuration entries when behavior needs to be tuned or reused.
- Update user-facing docs only when behavior changes; otherwise leave them untouched to preserve history.

### Logging & telemetry
- Route new metrics through the established logging surfaces so terminal, CSV, and external loggers stay aligned.
- Keep console output resilient to non-TTY environments by opting into ANSI/colour only when supported.
- Favor structured payloads over ad-hoc prints; let higher-level loggers handle presentation.
- Treat metric keys as namespaced `<train|val|test>/<metric>` and use `utils.metrics_config.metrics_config` helpers for construction/parsing; do not split strings inline.
- Validate metrics against configured bounds and delta rules at the logging boundary; fail fast on violations.

### Error handling
- **Fail fast and fail loud**: Never catch and silence exceptions. Let errors surface immediately and clearly.
- **No backwards compatibility**: When making changes, break existing code rather than maintaining compatibility. Force users to update their code.
- **No safeguards or defensive programming**: Don't add try/catch blocks, validation layers, or compatibility shims. Let the system fail if inputs are wrong.
- **Explicit error propagation**: Never use broad exception handlers (`except Exception:`) or bare `except:` clauses. Catch only specific, expected exceptions and re-raise with context.
- **Assert liberally**: Use assertions for all preconditions, postconditions, and invariants. Let the program crash if assumptions are violated.
- **No graceful degradation**: If something can't work correctly, fail immediately rather than falling back to partial functionality.

### Testing expectations
- Add or adjust tests alongside the functionality they cover, mirroring existing test layouts.
- Prefer deterministic fixtures or lightweight stubs to keep tests fast and reliable.
- Run the relevant test subset after meaningful behavior changes whenever practical.

### Performance considerations
- Reuse buffers and avoid per-step allocations in hot paths to keep training runs efficient.
- Be mindful of device transfers; operate on tensors where they already live and lean on vectorized operations.
- Hoist immutable lookup sets/dicts to module scope (e.g., `_ALLOWED_NAMESPACES = frozenset({...})`) to avoid per-call allocations.

### Collaboration hygiene
- Keep diffs focused and reviewable, with commit messages that describe the root cause and the fix.
- Where new behavior alters developer workflows, add a brief note to the appropriate doc (`README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, or task playbooks).
- Verify linting/tests before submitting changes when the scope justifies it, and flag any known gaps for reviewers.
