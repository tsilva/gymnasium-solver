# Dependency health check

## Goal
Keep runtime and development dependencies secure, reproducible, and aligned with project expectations.

## Steps
1. Inventory declared dependencies from `pyproject.toml` (runtime, optional extras, and dev tools).
2. List installed versions with `uv pip list | cat` (or `pip list | cat` when `uv` is unavailable) and capture the current `python --version`.
3. Identify outdated or vulnerable packages using `uv pip list --outdated | cat`; for each candidate, review changelogs or local release notes when available offline.
4. Decide on safe upgrades: prefer patch/minor bumps; flag major upgrades separately if they alter APIs or interoperability (e.g., Gymnasium v0→v1 transitions).
5. Apply upgrades (`uv pip install --upgrade <package>` or edit `pyproject.toml`) and regenerate lockfiles (`uv lock`) when changes are kept.
6. Validate dependency graph integrity with `uv pip check` (fallback: `pip check`) and resolve any conflicts.
7. Run the full test suite (`pytest -q`) and, when practical, a short sanity training run (`python train.py Bandit-v0:ppo -q`) to ensure learning still converges.
8. Update documentation when install instructions change (e.g., new optional extras) and summarize version shifts plus any follow-up tasks.

## Notes
- Respect pinned versions required by upstream tooling; avoid loosening pins without verifying compatibility downstream (gym wrappers, Lightning, SB3).
- If a package cannot be upgraded due to regressions, document the blockade and capture a follow-up task.
- Never leave the environment in a broken state; rollback partial upgrades if tests fail and no fix is provided.
- When a dependency has a security advisory, note the CVE or bulletin number in the summary and track remediation status.
