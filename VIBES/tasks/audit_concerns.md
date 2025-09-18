# Find separation of concerns issues

## Goal
Identify where responsibilities bleed across layers or modules and document precise remediation plans. Do not change code as part of this task. Capture each proposed fix as a `REFACTOR:` item in `TODO.md` aligned to the architecture guide.

## Steps
1. Review the current layering and allowed dependencies in `VIBES/ARCHITECTURE_GUIDE.md` and `README.md` (e.g., presentation vs. orchestration vs. domain/core vs. infrastructure). Note what belongs where and which directions are allowed for imports.
2. Map the codebase to those boundaries (directories, packages, or components) without assuming any one layout. Focus on high-churn or central modules first.
3. Scan for common boundary leaks, such as:
   - Input/Output in core logic (console, files, network, or external services in otherwise pure modules).
   - Configuration/CLI parsing or environment access mixed into runtime execution paths.
   - Logging/metrics entangled with domain logic instead of routed through adapters/hooks.
   - Cross-layer imports that invert intended dependency direction.
   - Tight loops doing blocking IO or side effects that belong in adapters.
4. Validate each suspected issue by tracing control flow and capturing a short note: where it occurs, why it violates the intended boundary, and the smallest viable correction.
5. Draft minimal realignment proposals that preserve public APIs where possible (e.g., extract a helper, introduce an adapter/port, move code to the right module, or add a thin interface to invert a dependency). Do not modify code.
6. For each confirmed issue, add a `REFACTOR:` entry to `TODO.md` with:
   - A concise problem statement and why it violates boundaries.
   - The minimal corrective action and target files/modules.
   - Any risks/assumptions and a rough size estimate (S/M/L).
7. Do not implement changes or modify tests/docs during this audit. Skip running test suites beyond lightweight static analysis.
8. Ensure the `REFACTOR:` items are clearly actionable and reference file paths/symbols as anchors.

## Search tips
- Use fast repo-wide search (e.g., `rg`) tuned to your stack to spot IO, CLI/config, and network usage in inappropriate layers.
- Prefer category searches over specific libraries so guidance stays future-proof. Tailor patterns to the libraries currently in use.

## Notes
- No code changes in this task. Produce `REFACTOR:` items only.
- Prefer improving existing abstractions over adding new global helpers; clarity beats cleverness (to be applied when executing the refactors).
- If a violation implies a larger redesign, scope it clearly and capture a follow-up `REFACTOR:` (or `TASK:` when broader) rather than landing a partial move.
