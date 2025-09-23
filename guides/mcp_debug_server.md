# MCP Debug Server

This helper server captures the shortcuts I kept wishing for while debugging the
`ALE-Pong-v5_objects` PPO run:

- **Run triage was manual** – I had to tab through dozens of `runs/<id>/` folders,
  inspect `config.json`, skim `metrics.csv`, and eyeball `run.log` just to learn
  whether a run ever left -20 reward or if KL/clip values were drifting.
- **Metric inspection was noisy** – grabbing the tail of `train/ep_rew_mean` or
  `train/approx_kl` meant opening large CSVs and manually scanning for the last
  non-empty entry.
- **Env instrumentation required ad-hoc scripts** – validating that OCAtari was
  surfacing `Ball`, `Player`, and `Enemy` objects (and that the feature vector
  stayed consistent) meant spinning up custom Python one-offs.

To make the next session faster and cheaper in tokens, I wrapped those chores in
an MCP server that speaks a tiny JSON-RPC surface over stdio.

## Running the server

```bash
# From repo root (uses current directory as the workspace root)
python -m mcp.debug_server
```

The process reads newline-delimited JSON from stdin and writes responses to
stdout, making it easy to integrate with an MCP-compatible agent or script.

### Core protocol

- List capabilities:
  ```json
  {"id": 1, "method": "list_tools"}
  ```

- Call a tool:
  ```json
  {
    "id": 2,
    "method": "call_tool",
    "params": {
      "name": "summarize_run",
      "args": {"run_id": "sdpskiii"}
    }
  }
  ```

### Available tools

| Tool | Purpose | Key fields |
| --- | --- | --- |
| `list_runs` | Recent runs with coarse stats (filterable by `project_id`). | `runs[*].train`, `runs[*].val`, `total_timesteps` |
| `summarize_run` | Single-run snapshot of reward, KL, and clip stats plus config highlights. | `train`, `val`, `total_timesteps` |
| `metrics_tail` | Last _N_ samples of any metric column (default `train/`). | `samples`, `stats` |
| `config_slice` | `config.json` with optional key filtering for lightweight inspection. | `config` |
| `probe_objects` | Steps an OCAtari env for quick object/feature sanity checks. | `snapshots`, `unique_categories` |

Each tool returns compact JSON so the agent only needs to ship structured data
instead of raw logs. See the docstrings in `mcp/debug_server.py` for parameter
schemas.

## Example snippets

- **List recent Pong-object runs**
  ```json
  {
    "method": "call_tool",
    "id": 3,
    "params": {
      "name": "list_runs",
      "args": {"project_id": "ALE-Pong-v5_objects", "limit": 3}
    }
  }
  ```

- **Inspect how KL behaved near the end**
  ```json
  {
    "method": "call_tool",
    "id": 4,
    "params": {
      "name": "metrics_tail",
      "args": {"run_id": "sdpskiii", "metric": "approx_kl", "stage": "train", "limit": 5}
    }
  }
  ```

- **Verify OCAtari object stream**
  ```json
  {
    "method": "call_tool",
    "id": 5,
    "params": {
      "name": "probe_objects",
      "args": {"env_id": "ALE/Pong-v5", "n_steps": 8, "include_features": true}
    }
  }
  ```

## Extending the toolset

- Drop new helpers into `MCPDebugServer._register_tools` with clear JSON schemas.
- Prefer returning aggregated values (means, min/max, etc.) rather than raw rows
  to keep responses compact.
- If a helper needs repo context (config paths, run artefacts, env wrappers),
  `repo_root` is injected at construction and defaults to `cwd`.

This server should remove most of the repetitive file peeking and ad-hoc env
scripts from the next debugging session so we spend tokens on analysis instead
of plumbing.
