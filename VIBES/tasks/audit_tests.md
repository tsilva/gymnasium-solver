# Run all test suites

## Goal
Exercise every automated test target and document failures or flakiness with precise remediation tasks. Do not change code or tests as part of this task. Capture each finding as a `TEST:` and/or `BUG:` item in `TODO.md` with exact file paths and reproduction details.

## Steps
1. Run the full suite (`pytest -q` unless a project-specific command is documented); capture the full failure output for each failing test case.
2. For each failure, determine whether the test expectation is wrong or product code has regressed. Do not patch code. Collect a minimal reproduction, suspected root cause, and impacted files/symbols.
3. Add `TEST:` items for test-side fixes (fixtures/assertions/helpers) and `BUG:` items for product defects. Each entry should include:
   - Exact file paths and symbols (e.g., `tests/test_metrics_table_logger.py::test_sorted_metrics`).
   - Proposed minimal change and acceptance criteria.
4. Optionally re-run targeted tests to confirm reproductions remain consistent. Do not implement fixes during this audit.
5. Summarize the work: list flagged bugs and test issues, and link to the corresponding `TODO.md` entries.

## Notes
- No code or test edits in this task—produce `TEST:`/`BUG:` items only.
- Preserve existing test structure in follow-ups; prefer targeted updates over rewrites.
- Do not paper over real failures—include stack traces, configs, and environment details in the TODO entries.
- Keep tests deterministic: pin random seeds and mock external services locally when needed (document these in acceptance criteria).
