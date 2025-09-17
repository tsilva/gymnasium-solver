# Find bugs

## Goal
Expose and resolve functional defects before they reach users or downstream pipelines.

## Steps
1. Establish a clean baseline: run `pytest -q` (and any smoke scripts such as `python train.py Bandit-v0:ppo -q`) to surface existing failures; capture stack traces and impacted modules.
2. Mine signals from `TODO.md`, recent commits, and failing CI runs (if available); prioritize areas with recent churn or high-impact functionality (training loop, rollout collection, environment wrappers, logging).
3. Reproduce each suspected defect deterministically—craft the smallest script or test that triggers the issue, locking seeds and inputs where possible.
4. Diagnose root cause by tracing data flow through the relevant functions/classes; inspect adjacent modules to avoid treating symptoms.
5. Implement the minimal fix that addresses the underlying bug, preserving existing style and avoiding unrelated refactors.
6. Add or update regression tests that fail without the fix and pass afterward; keep test runtime acceptable.
7. Re-run `pytest -q` (plus targeted scripts if needed) to confirm the fix and guard against regressions.
8. Document the discovered bug, fix rationale, and follow-up work (if any) in the task summary or `TODO.md`.

## Notes
- Favor high-signal logging or assertions during investigation but remove any temporary instrumentation before finalizing the fix.
- If a bug cannot be resolved quickly, ensure the failing reproduction and analysis are captured for future agents.
