# Find separation of concerns issues

## Goal
Identify where responsibilities bleed across layers or modules, then realign code to clear, intentional boundaries defined by the architecture guide.

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
5. Propose minimal realignments that preserve public APIs where possible: extract a helper, introduce an adapter/port, move code to the right module, or add a thin interface to invert a dependency.
6. Implement incremental changes and update impacted tests/docs. Keep diffs small and avoid opportunistic refactors unrelated to the violation.
7. Run `pytest -q` (and any quick smoke scripts) to confirm behavior remains correct.
8. Record findings and actions (or deferrals with rationale) in `TODO.md` or a task report.

## Search tips
- Use fast repo-wide search (e.g., `rg`) tuned to your stack to spot IO, CLI/config, and network usage in inappropriate layers.
- Prefer category searches over specific libraries so guidance stays future-proof. Tailor patterns to the libraries currently in use.

## Notes
- Prefer improving existing abstractions over adding new global helpers; clarity beats cleverness.
- If a violation implies a larger redesign, scope it clearly and create a follow-up task rather than landing a partial move.
