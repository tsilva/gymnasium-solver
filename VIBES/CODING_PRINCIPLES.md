## Coding Principles

This companion to `VIBES/ARCHITECTURE_GUIDE.md` captures the high-level principles we follow when writing or refactoring code. For architecture, data flow, and subsystem ownership, defer to `VIBES/ARCHITECTURE_GUIDE.md`.

### General principles
- Respect the config-first workflow: surface behavior through configuration or data models instead of hard-coded branches.
- Compose before reinventing: reach for existing helpers, registries, and extension points so new work integrates cleanly.
- Keep edits scoped: change only what the task requires, and avoid opportunistic refactors that mix concerns.

### Python style
- Use type hints consistently, including container generics and optional markers where intent matters.
- Provide short docstrings for non-trivial helpers; reserve inline comments for explaining intent or edge cases.
- Prefer early returns and guard clauses to reduce nesting and keep control flow readable.
- When grouping related fields, reach for small dataclasses or named tuples rather than loosely structured dicts.

### Documentation & configuration
- Centralize file IO through the shared utilities and keep encoding UTF-8 by default.
- Avoid magic values—introduce constants or configuration entries when behavior needs to be tuned or reused.
- Update user-facing docs only when behavior changes; otherwise leave them untouched to preserve history.

### Logging & telemetry
- Route new metrics through the established logging surfaces so terminal, CSV, and external loggers stay aligned.
- Keep console output resilient to non-TTY environments by opting into ANSI/colour only when supported.
- Favor structured payloads over ad-hoc prints; let higher-level loggers handle presentation.

### Error handling
- Fail loudly on invalid state with targeted assertions or specific exceptions; do not swallow errors.
- Catch exceptions only when the API contract expects it, and re-raise with added context when helpful.
- Derive deterministic seeds from shared configuration so reproducibility is the default.

### Testing expectations
- Add or adjust tests alongside the functionality they cover, mirroring existing test layouts.
- Prefer deterministic fixtures or lightweight stubs to keep tests fast and reliable.
- Run the relevant test subset after meaningful behavior changes whenever practical.

### Performance considerations
- Reuse buffers and avoid per-step allocations in hot paths to keep training runs efficient.
- Be mindful of device transfers; operate on tensors where they already live and lean on vectorized operations.

### Collaboration hygiene
- Keep diffs focused and reviewable, with commit messages that describe the root cause and the fix.
- Where new behavior alters developer workflows, add a brief note to the appropriate doc (`README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, or task playbooks).
- Verify linting/tests before submitting changes when the scope justifies it, and flag any known gaps for reviewers.
