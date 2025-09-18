# Update README.md

## Goal
Keep the README concise, accurate, and aligned with the repository's current features and workflows.

## Steps
1. Collect inputs: scan `TODO.md`, `AGENTS.md`, `VIBES/ARCHITECTURE_GUIDE.md`, and recent commits for items that should surface in the README.
2. Audit the existing sections (Highlights, Quickstart, Configs, Environment wrappers, Publishing, etc.) against the latest code and tooling to spot stale assertions or missing features.
3. Verify command snippets still work (`uv sync`, `python train.py ...`, `python inspector.py ...`, publishing flow) or flag them for follow-up if they require unavailable services.
4. Draft minimal, targeted edits that update the relevant subsections while preserving the current tone, emoji usage, and formatting cadence.
5. Cross-check cross-references (paths, task names such as `run task: find separation of concerns`) so automation and docs stay in sync.
6. Proofread for typos, lint Markdown if tooling exists, and double-check rendered output (e.g., via a Markdown preview) when feasible.
7. Summarize the README changes, noting any deferred follow-up work.

## Notes
- Prefer factual updates over marketing copy; defer larger restructuring to a dedicated doc rewrite task.
- Keep code blocks truthful to actual CLI usage; avoid speculative flags or options.
- Only touch sections that require changes—drive-by rewrites break the minimal-diff rule.
