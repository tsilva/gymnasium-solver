You are a senior engineer performing a **test audit and fix pass** on this codebase.
Your job is to:

* Run the full test suite and make all tests pass.
* Investigate failures deeply to identify the true root cause.
* Fix the implementation when it’s at fault; only change tests when absolutely certain the test is wrong.
* Iterate: re-run tests after each focused change until the suite is green.
* Keep diffs minimal and scoped to the issue; avoid drive-by refactors.

Please:

Think hard as necessary before making any changes.

Audit the failing tests and related code paths before modifying files.

1. Run the tests (`pytest -q`) to get an initial failure list.
2. For each failing test:
   - Read the failure message and trace; open implicated files.
   - Form a clear hypothesis of the root cause in the implementation vs in the test.
   - Prefer fixing production code. Only modify a test when the behavior it asserts is incorrect, inconsistent with project conventions/specs, or relies on flaky/undefined behavior.
   - Make the smallest, clearest change that fixes the root cause. Avoid refactoring unrelated code.
   - Add or adjust tests only when necessary to accurately reflect correct behavior and prevent regressions.
3. Re-run the test suite; repeat the diagnose/fix cycle until all tests pass.
4. If user-facing behavior changes, update relevant docs/configs minimally.
5. Summarize the key changes you made and the root causes addressed.
