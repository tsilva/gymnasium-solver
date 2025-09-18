# Codify principles from uncommitted diff

## Goal
Infer the developer’s current coding preferences from uncommitted changes and reflect them in `VIBES/CODING_PRINCIPLES.md` with small, targeted edits. This reduces recurring “AI slop” cleanups by making intent explicit.

## Steps
1. Prep context
   - Read `README.md`, `VIBES/ARCHITECTURE_GUIDE.md`, and `VIBES/CODING_PRINCIPLES.md` end‑to‑end to align tone and sections.
   - Skim `AGENTS.md` for any repo‑specific rules that affect editing docs.
   - Confirm this task explicitly authorizes edits to `VIBES/CODING_PRINCIPLES.md` (it does); still keep diffs minimal and scoped.

2. Gather diffs (both staged and unstaged)
   - `git status | cat` to verify there are local modifications.
   - `git diff --name-only | cat` and `git diff -U0 | cat` for precise changed lines.
   - If there are staged changes: `git diff --staged --name-only | cat` and `git diff --staged -U0 | cat`.
   - Ignore obviously generated or vendored files. Focus on: `agents/`, `utils/`, `gym_wrappers/`, `trainer_callbacks/`, `loggers/`, `scripts/`, `config/`, `tests/`.

3. Infer intent from patterns
   - For each touched file, note high‑level intent behind edits. Classify into buckets: naming, typing, structure, logging, error handling, configuration, tests, docs, metrics, performance, CLI ergonomics.
   - Use these heuristics to derive candidate principles (examples):
     - Type hints added or tightened → “Use type hints consistently (including container generics).”
     - Replaced `print` with logger calls → “Favor structured logging over ad‑hoc prints.”
     - Broad `except:` replaced with specific → “Catch only specific, expected exceptions; re‑raise with context.”
     - Constants/config pulled from code into YAML → “Respect the config‑first workflow; avoid hard‑coded branches.”
     - Long function split into helpers → “Prefer small, composable helpers and early returns.”
     - Docstrings/comments added for tricky paths → “Document non‑trivial helpers; explain intent/edge cases briefly.”
     - Test coverage added for changed logic → “Add/adjust tests alongside behavior changes.”
     - Metrics unified via existing logger → “Route metrics through established logging surfaces.”
     - Renames toward clarity → “Choose descriptive identifiers; avoid cryptic names.”
     - Removed duplicate logic → “DRY: extract and reuse helpers; avoid copy‑paste variants.”
   - Capture a short evidence note for each candidate (file path and a brief before/after summary).

4. Reconcile with existing principles
   - Open `VIBES/CODING_PRINCIPLES.md` and find the closest section: General, Python style, Documentation & configuration, Logging & telemetry, Error handling, Testing expectations, Performance, Collaboration hygiene. Create a new section only if necessary.
   - If a matching principle exists, refine or strengthen it instead of adding a near‑duplicate. If a principle is contradicted by repeated changes, update or remove it—only when the diff shows a consistent shift.
   - Keep bullets one‑line, imperative, and high‑signal. Avoid repeating architecture details (those belong in `VIBES/ARCHITECTURE_GUIDE.md`).

5. Draft minimal edits safely
   - Draft the proposed doc in `VIBES/CODING_PRINCIPLES.proposed.md` first, mirroring the original structure.
   - Compare against the current file and trim to the smallest set of additions/edits/removals needed to encode the observed intent.
   - If the inferred intent is ambiguous or one‑off, do not update the canon. Instead, write a short note to `TODO.md` explaining the observation and recommendation.

6. Apply and validate
   - Apply the minimal changes to `VIBES/CODING_PRINCIPLES.md` (preserve formatting; don’t reflow unrelated lines).
   - Review with `git diff VIBES/CODING_PRINCIPLES.md | cat` and ensure only intended bullets changed.
   - Optionally run `pytest -q` to ensure no incidental breakages if code was also modified in this session.
   - Delete the `*.proposed.md` scratch file if it was used.

7. Summarize
   - Record 3–7 bullet points of the newly codified principles with brief evidence (e.g., `utils/foo.py:42 → replaced broad except with ValueError`).
   - Include a one‑line root‑cause statement (e.g., “Recurring refactors around logging and exception hygiene indicated a stronger preference for structured logs and specific exceptions.”).

## Notes
- Non‑destructive default: if no meaningful patterns emerge from the current diff, make no edits and capture findings in `TODO.md` instead.
- Keep edits localized: do not introduce sweeping reformatting, section renaming, or stylistic churn.
- Match voice and structure of the existing document; avoid duplicating content from the Architecture Guide.
- Do not modify `VIBES/tasks/*.md` while executing this task. Only edit `VIBES/CODING_PRINCIPLES.md` when intent is clear.

