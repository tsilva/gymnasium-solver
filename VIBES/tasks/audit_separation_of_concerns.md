# Find separation of concern violations

## Goal
Identify modules where responsibilities bleed across layers and refactor them into clearer boundaries.

## Steps
1. Review architecture guidance in `VIBES/ARCHITECTURE_GUIDE.md` and recent feature work to understand intended layering (agents, utils, wrappers, scripts).
2. Scan high-churn modules (`agents/`, `utils/`, `trainer_callbacks/`, `gym_wrappers/`) looking for telltale smells: UI/CLI code inside core logic, file IO inside tight training loops, cross-package imports that violate dependency direction, or mixing config parsing with runtime execution.
3. Use `rg` to locate suspicious keywords (e.g., `print(`, `os.environ`, `argparse`, `requests`) inside modules that should remain pure or model-focused.
4. For each candidate violation, confirm the breach by tracing control flow and documenting why the responsibility is misplaced (e.g., agent classes mutating filesystem paths, wrappers managing logging, utils reaching into CLI state).
5. Propose the minimal realignment: extract helpers, introduce a dedicated utility, or move logic to the appropriate layer while keeping public APIs stable.
6. Implement the adjustment incrementally and update tests/docs impacted by the move; avoid broad rewrites unless strictly necessary.
7. Run `pytest -q` and any relevant smoke scripts to ensure behavior remains correct after refactoring.
8. Summarize each violation and the corrective action (or the rationale for deferring it) in the task report or `TODO.md`.

## Notes
- Prefer improving existing abstractions over inventing new global helpers; clarity beats cleverness.
- When a violation is entrenched and requires a larger redesign, describe the scope and create a follow-up task rather than partially moving code.
