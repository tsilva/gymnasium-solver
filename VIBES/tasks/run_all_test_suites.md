# Run all test suites

## Goal
Exercise every automated test target, repair flaky or outdated tests, and surface genuine product bugs without masking them.

## Steps
1. Run the full suite (`pytest -q` unless a project-specific command is documented); capture the full failure output for each failing test case.
2. For each failure, investigate whether the test expectation is wrong or the product code has regressed. When behavior appears correct and the test is stale or brittle, move to Step 3; otherwise collect a minimal reproduction, note the suspected root cause, and flag the bug to the user instead of patching production code.
3. Fix broken tests by adjusting fixtures, assertions, or localized helpers under `tests/`; keep shims colocated with the affected tests (e.g., `tests/helpers/`) and never modify the main source directories to accommodate tests.
4. Re-run only the affected tests to confirm the fix, then re-run the whole suite to ensure it is clean.
5. Summarize the work: list remaining flagged bugs, document any skipped fixes, and call out new test-side helpers that were introduced.

## Notes
- Preserve existing test structure; prefer targeted updates over rewrites.
- If multiple suites or optional extras exist, run them all (integration, lint, property tests) and record their outcomes.
- Do not paper over real failures—provide stack traces, configs, and environment details when escalating bugs to the user.
- Keep tests deterministic: pin random seeds and mock external services locally when needed.
- Remove any temporary instrumentation once a test passes to avoid noisy diffs.
