You are a senior engineer performing a deep codebase review focused on the accuracy and cohesion of INTERNALS.md.

Your job is to:

* Read repository documentation and core modules (e.g., README.md, INTERNALS.md, top-level scripts, key packages) to build an accurate mental model.
* Map the actual control flow and cross-module interactions (e.g., entrypoints → configuration → initialization → core pipelines → persistence/logging/reporting).
* Identify mismatches, outdated claims, or ambiguous statements in INTERNALS.md relative to the current code and tests.
* Propose concise improvements and then apply minimal edits to INTERNALS.md to correct inaccuracies and clarify behavior.

Please:

Think carefully and verify behaviors in code/tests before changing docs. Favor precise, minimal diffs that improve clarity. Match existing formatting and tone.

Generic checklist (adapt as applicable to the project):

1) Entrypoints & CLI
   - Current commands/flags and how configuration is supplied.
   - Any runtime prerequisites or environment variables.

2) Configuration
   - How configurations are loaded/merged (files, env, CLI) and defaults.
   - Any schedule/templating conventions; computed or derived fields.

3) Initialization & resources
   - How core services/environments/resources are created and wired.
   - Normalization/preprocessing steps and where they are toggled.

4) Core flow
   - Main loops/pipelines (training, processing, serving, etc.).
   - Data loading/streaming patterns and batching semantics.
   - Scheduling/decay mechanisms tied to progress or time.

5) Evaluation/monitoring & persistence
   - Validation/evaluation cadence and any recording/capture steps.
   - Run/artifact layout, checkpointing/versioning, and resume mechanics.
   - Metrics/printing rules and highlight/threshold logic if applicable.

Deliverables:

1. Make targeted edits to INTERNALS.md so it accurately reflects the current code behavior (avoid speculative features).
2. Keep style and structure consistent; fix small formatting issues only when adjacent to your edits.
3. Summarize the changes you made and the reasoning.

Perform your audit/edit on: INTERNALS.md
