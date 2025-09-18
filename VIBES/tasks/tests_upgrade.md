# Strengthen test suite

## Goal
Increase confidence in critical behaviors by adding or improving automated tests for high-risk modules.

## Steps
1. Establish a baseline: run `pytest -q` and note existing failures, slow tests, and skipped modules.
2. Map uncovered areas by comparing `tests/` against high-complexity code in `agents/`, `utils/`, `gym_wrappers/`, and `trainer_callbacks/` (use `radon cc` or manual inspection when tooling is unavailable).
3. Prioritize targets that impact training loops, rollout collection, and environment wrappers; capture recent regressions or TODOs first.
4. Design the test before modifying production code—state the expected behavior and reproduce a failing case when the current implementation is wrong or untested.
5. Implement the minimal test (and accompanying fix if needed), keeping fixtures consistent with existing test patterns and avoiding broad refactors.
6. Re-run `pytest -q`; ensure new tests fail without the fix when feasible and pass afterward.
7. Document any notable coverage gains or remaining gaps in the task summary, linking to follow-up work if large areas remain untested.

## Notes
- Use deterministic seeds (`Config.seed`) and lightweight env variants to keep tests fast and reproducible.
- Prefer unit tests over full training runs unless integration coverage is required to reproduce an issue.
- Update `README.md` or `VIBES/ARCHITECTURE_GUIDE.md` only if public-facing expectations change; otherwise keep documentation untouched.
- Capture coverage deltas when feasible (`pytest --cov` or manual accounting) to show progress and highlight remaining blind spots.
