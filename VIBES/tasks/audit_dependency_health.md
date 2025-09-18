# Dependency health check

## Goal
Assess and document the state of runtime and development dependencies. Do not change the environment or project files as part of this task. Capture each proposed change as an actionable item in `TODO.md` (use `TASK:`/`TODO:` with exact file paths like `pyproject.toml`) and include acceptance criteria.

## Steps
1. Inventory declared dependencies from `pyproject.toml` (runtime, optional extras, and dev tools).
2. List installed versions with `uv pip list | cat` (or `pip list | cat` when `uv` is unavailable) and capture the current `python --version`.
3. Identify outdated or vulnerable packages using `uv pip list --outdated | cat`; for each candidate, note current vs. latest and constraints in `pyproject.toml`.
4. Propose safe upgrades: prefer patch/minor bumps; flag major upgrades separately if they alter APIs or interoperability (e.g., Gymnasium v0→v1 transitions). Do not make changes.
5. For each agreed candidate, add a `TASK:` entry to `TODO.md` specifying:
   - Exact edits to `pyproject.toml` (paths/sections) and any extras.
   - Lockfile refresh steps (e.g., `uv lock`).
   - Post-upgrade validation: `uv pip check`, `pytest -q`, and a brief sanity training run.
6. Capture any conflicts or security advisories and include CVE/bulletin references in the TODO entry.
7. Do not modify dependencies in this audit; ensure all tasks have clear acceptance criteria and rollback guidance.

## Notes
- No environment or file edits in this task—produce actionable `TASK:`/`TODO:` items only.
- Respect pinned versions required by upstream tooling; note compatibility considerations in follow-ups (gym wrappers, Lightning, SB3).
- If a package cannot be upgraded due to regressions, record the blockade and link to issues/changelogs when available offline.
- Include CVEs or advisories and specify verification commands in acceptance criteria.
