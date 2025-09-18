# Architecture Guide maintenance audit

## Goal
Ensure `VIBES/ARCHITECTURE_GUIDE.md` stays accurate, high-signal, and useful so agents can understand critical architecture and workflows without excess code spelunking.

## Steps
1. Read the latest `VIBES/ARCHITECTURE_GUIDE.md` (currently at `VIBES/ARCHITECTURE_GUIDE.md`) end-to-end to capture its canonical structure, sections, and assumptions.
2. Inventory recent code changes: review `git log --since` for the target window (and `git status` for in-flight work), skim highlighted modules (`agents/`, `utils/`, `gym_wrappers/`, `trainer_callbacks/`, `scripts/`, `VIBES/`), and note new subsystems or major refactors.
3. Compare documented flows (training entry point, config loading, env construction, logging, evaluation, publishing) against current implementations; confirm APIs, defaults, and control flow still match.
4. Identify missing or stale details that would materially help an agent complete tasks faster (e.g., new helper modules, changed directories, updated CLI flags, altered lifecycle hooks, deprecations).
5. Draft concise additions or edits to `VIBES/ARCHITECTURE_GUIDE.md`, preserving tone and formatting; focus on high-signal summaries, not exhaustive re-documentation.
6. Validate links, references, and code identifiers—open referenced files to double-check names and confirm guidance aligns with actual behavior.
7. Run `git diff VIBES/ARCHITECTURE_GUIDE.md` to review changes, ensure no accidental noise, and update related docs (`README.md`, `AGENTS.md`) only if their statements also became stale.
8. Record the audit outcome (date, touched sections, unresolved gaps) in the task summary or `TODO.md` for traceability.

## Notes
- Keep the document actionable: prefer bullet summaries, call out gotchas, and avoid duplicating `README.md` marketing content.
- If a discovery requires deep follow-up (e.g., missing design docs), record it in `TODO.md` or create an additional task rather than bloating `VIBES/ARCHITECTURE_GUIDE.md`.
- When no edits are needed, still log the audit date and findings in the task summary for traceability.
