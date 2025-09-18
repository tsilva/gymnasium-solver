# Config consistency audit

## Goal
Validate that configuration files, registries, and defaults remain coherent so agents and users receive predictable behavior.

## Steps
1. Enumerate environment configs under `config/environments/` and note their declared variants (`ppo`, `reinforce`, etc.).
2. Check each config against `utils/config.py` expectations: confirm required keys exist, schedules are well-formed, and derived values (e.g., fractional `batch_size`) make sense.
3. Verify that wrapper IDs referenced in configs are registered in `gym_wrappers/__init__.py` and that parameter names match constructor signatures.
4. Inspect defaults in `agents/__init__.py`, `agents/base_agent.py`, and `train.py` to ensure they align with documented defaults (e.g., fallback config `Bandit-v0:ppo`).
5. Run targeted dry-run loads: execute `python - <<'PY'` snippets to call `load_config(env, variant)` for each environment, asserting that the resulting `Config` has expected defaults (e.g., positive `n_steps`, valid wrapper kwargs) and raises no exceptions.
6. For any inconsistencies, choose the minimal fix—update YAML, adjust loader validation, or correct documentation.
7. Re-run affected config loads and `pytest -q` (especially config-centric tests) to confirm everything passes, then summarize the reconciled issues.

## Notes
- Avoid mass reformatting of YAML; limit changes to intentional corrections.
- When removing or renaming configs, update associated docs (`README.md`, guides) and automation scripts to prevent broken references.
- Confirm ancillary registries (e.g., `config/metrics.yaml`, wrapper registries, sweeps) stay in sync whenever config identifiers change.
