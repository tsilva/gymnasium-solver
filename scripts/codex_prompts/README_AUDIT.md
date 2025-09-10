You are a senior engineer performing a thorough codebase review and README improvement pass.

Your job is to:

- Map the architecture end-to-end (entry points, core modules, extension points, background services).
- Verify README claims against the actual code and CLIs (commands, flags, defaults, config keys, paths).
- Identify missing, misleading, or outdated documentation.
- Propose a concise structure for README that prioritizes onboarding and accuracy.
- Update README.md with minimal diffs that fix inaccuracies and enhance clarity.

Please:

Think carefully before editing; read the repository holistically first.

Review at minimum:
- Top-level docs (e.g., README.md, CONTRIBUTING.md, INTERNALS/ARCHITECTURE docs if present).
- Primary entry points (CLI apps, scripts, binaries, services) and how they are invoked.
- Core packages/modules and any plugin/extension systems (wrappers, hooks, callbacks, etc.).
- Configuration sources (YAML/TOML/JSON files, environment variables, CLI flags).

Then:
1. Inventory core features, supported platforms/domains, and primary workflows.
2. Outline key data/control flows (e.g., input → processing → output; lifecycle hooks).
3. Validate commands and configuration semantics (actual CLI signatures, config keys/defaults, environment variables).
4. Draft targeted README improvements: Installation, Quickstart, Configuration, Usage examples, Extensibility, Project structure, Testing, Troubleshooting.
5. Apply minimal, surgical edits to README.md; preserve tone/style while fixing inaccuracies and gaps.
6. Summarize the key changes and the mismatches you corrected.

Perform your changes on:

- README.md (primary)
- Internal docs (e.g., INTERNALS.md, ARCHITECTURE.md, or equivalents) only if cross-referenced behavior changed.

Return the updated files and a brief summary of corrections.
