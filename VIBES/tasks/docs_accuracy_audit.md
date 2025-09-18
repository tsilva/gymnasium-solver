# Documentation accuracy audit

## Goal
Ensure public and internal documentation reflects the current behavior, tooling, and workflows of the repository.

## Steps
1. Build a doc inventory: list high-visibility files (`README.md`, `AGENTS.md`, `VIBES/ARCHITECTURE_GUIDE.md`, `guides/*.md`, `TODO.md`).
2. Verify setup instructions by replaying critical commands (`uv sync`, `pip install -e .`, `python train.py Bandit-v0:ppo -q`) in a clean environment when feasible; note any surprises or missing prerequisites.
3. Cross-check feature descriptions (algorithms, configs, wrappers, publishing flows) against the implementation in `agents/`, `utils/config.py`, `gym_wrappers/`, and scripts under `scripts/`.
4. Flag stale or contradictory statements (e.g., renamed folders, relocated helper files like `VIBES/tasks/`, deprecated CLI flags, mismatched defaults) and draft corrected wording.
5. Apply minimal edits to the affected docs, preserving formatting guidelines from the surrounding text.
6. Run spell-check or linting if available (`codespell`, `mdformat --check`); otherwise manually review for typos and broken Markdown links.
7. Summarize the doc updates and capture any larger content gaps that require a separate, deeper rewrite.

## Notes
- Keep changelogs concise; avoid turning quick fixes into narrative blog posts.
- When unsure about an instruction, prefer marking it for follow-up instead of guessing—accuracy beats speculation.
- Double-check cross-document references (anchors, relative paths, task names) so automation that parses docs remains valid.
