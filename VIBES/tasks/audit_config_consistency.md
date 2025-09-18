# Config consistency audit

## Goal
Validate and document inconsistencies across configuration files, registries, and defaults. Do not change code or YAML as part of this task. Capture each confirmed issue as an actionable item in `TODO.md` (use `REFACTOR:` for loader/code changes, `TODO:`/`DOCS:` for config/doc updates) with precise file paths and keys.

## Steps
1. Enumerate environment configs under `config/environments/` and note their declared variants (`ppo`, `reinforce`, etc.).
2. Check each config against `utils/config.py` expectations: confirm required keys exist, schedules are well-formed, and derived values (e.g., fractional `batch_size`) make sense.
3. Verify that wrapper IDs referenced in configs are registered in `gym_wrappers/__init__.py` and that parameter names match constructor signatures.
4. Inspect defaults in `agents/__init__.py`, `agents/base_agent.py`, and `train.py` to ensure they align with documented defaults (e.g., fallback config `Bandit-v0:ppo`).
5. Run targeted dry-run loads (non-destructive): execute `python - <<'PY'` snippets to call `load_config(env, variant)` for each environment and capture exceptions or mismatches. Do not modify code.
6. For any inconsistencies, draft the minimal correction (update YAML key, adjust loader validation, or fix docs) and add a `REFACTOR:`/`DOCS:` item to `TODO.md` with:
   - Exact file paths and keys (e.g., `config/environments/cartpole.yaml:ppo.batch_size`).
   - Proposed change, rationale, and acceptance criteria (e.g., dry-run loads succeed, `pytest -q` passes config tests).
7. Do not implement changes during this audit; ensure each TODO entry is self-contained and actionable.

## Notes
- No code/YAML edits in this task—produce `REFACTOR:`/`DOCS:` items only.
- Avoid mass reformatting; keep proposed edits minimal and targeted.
- When removing or renaming configs is proposed, ensure follow-up TODOs cover docs (`README.md`, guides) and automation scripts.
- Confirm ancillary registries (e.g., `config/metrics.yaml`, wrapper registries, sweeps) are listed in follow-ups when identifiers change.
