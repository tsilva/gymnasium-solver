# Find bugs

## Goal
Identify and document functional defects with precise remediation plans. Do not change code as part of this task. Capture each confirmed issue as a `BUG:` item in `TODO.md` (and add a `REFACTOR:` follow-up when structural changes are implied).

## Steps
1. Establish a clean baseline: run `pytest -q` (and any smoke scripts such as `python train.py Bandit-v0:ppo -q`) to surface existing failures; capture stack traces and impacted modules. Do not apply fixes.
2. Mine signals from `TODO.md`, recent commits, and failing CI runs (if available); prioritize areas with recent churn or high-impact functionality (training loop, rollout collection, environment wrappers, logging).
3. Reproduce each suspected defect deterministically—craft the smallest script or test that triggers the issue, locking seeds and inputs where possible.
4. Diagnose root cause by tracing data flow through the relevant functions/classes; inspect adjacent modules to avoid treating symptoms.
5. Draft the minimal fix plan that addresses the underlying bug; do not modify code. Include suspected root cause, target files/symbols, and acceptance criteria.
6. For missing coverage or flaky tests, add a `TEST:` item describing the regression test to add/update and where it should live under `tests/`.
7. Add a `BUG:` entry to `TODO.md` that includes:
   - Reproduction snippet/command and observed vs. expected.
   - Exact file paths and symbols (e.g., `agents/base_agent.py:training_step`).
   - Minimal corrective action and risks.
8. Optionally validate that reproductions remain deterministic, but do not land fixes during this audit.

## Notes
- No code or test changes in this task—produce `BUG:` (and `TEST:`/`REFACTOR:` where applicable) items only.
- Favor high-signal logging or assertions during investigation, but remove or avoid committing temporary instrumentation.
- If a bug cannot be fully diagnosed, capture the best-available reproduction and analysis so the follow-up is actionable.
