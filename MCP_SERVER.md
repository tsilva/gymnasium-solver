# MCP Server for Gymnasium-Solver

An MCP (Model Context Protocol) server that provides tools for Claude Code to manage training runs and optimize reinforcement learning agents.

## Overview

This MCP server enables agent-guided optimization cycles where Claude Code can:
1. **Discover** available environments and configurations
2. **Launch** training runs with custom parameters
3. **Monitor** training progress and metrics
4. **Debug** issues by inspecting logs
5. **Compare** results across multiple runs
6. **Iterate** by tweaking configs and relaunching

## Installation

1. Install the MCP dependency:
```bash
uv pip install mcp
```

2. Configure Claude Code to use the MCP server by adding to your MCP settings:
```json
{
  "mcpServers": {
    "gymnasium-solver": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {},
      "description": "Training run management for gymnasium-solver"
    }
  }
}
```

3. Restart Claude Code to load the MCP server

## Available Tools

### Environment & Configuration Discovery

#### `list_environments`
List all available environment configurations.
- **Parameters:**
  - `filter` (optional): Filter string to match environment names
- **Returns:** List of environment IDs and their config files

#### `list_variants`
List available algorithm variants for an environment.
- **Parameters:**
  - `env_id`: Environment ID (e.g., "CartPole-v1")
- **Returns:** List of algorithm variants (e.g., ["ppo", "reinforce"])

#### `get_config`
Get full configuration for an environment:variant combination.
- **Parameters:**
  - `env_id`: Environment ID
  - `variant`: Algorithm variant
- **Returns:** Complete configuration dictionary

### Run Management

#### `list_runs`
List training runs with optional filtering.
- **Parameters:**
  - `env_filter` (optional): Filter by environment ID
  - `algo_filter` (optional): Filter by algorithm
  - `limit` (optional): Max runs to return (default: 20)
- **Returns:** List of runs with metadata

#### `get_run_info`
Get detailed information about a specific run.
- **Parameters:**
  - `run_id`: Run ID or "@last" for most recent
- **Returns:** Run details including config, checkpoints, metrics summary

#### `get_run_metrics`
Get metrics data from a training run.
- **Parameters:**
  - `run_id`: Run ID or "@last"
  - `metric_names` (optional): Specific metrics to retrieve
  - `limit` (optional): Limit number of rows
- **Returns:** Metrics data from metrics.csv

#### `get_run_logs`
Get log output for debugging.
- **Parameters:**
  - `run_id`: Run ID or "@last"
  - `lines` (optional): Number of lines from end (default: 100)
- **Returns:** Log content

### Training Control

#### `start_training`
Start a new training run.
- **Parameters:**
  - `config_id`: Config in "env:variant" format (e.g., "CartPole-v1:ppo")
  - `max_env_steps` (optional): Override max environment steps
  - `quiet` (optional): Run in quiet mode (default: true)
  - `wandb_mode` (optional): "online", "offline", or "disabled" (default: "disabled")
- **Returns:** Process info including PID

#### `stop_training`
Stop a running training process.
- **Parameters:**
  - `run_id`: Run ID to stop
- **Returns:** Success confirmation

#### `get_training_status`
Check if a training process is still running.
- **Parameters:**
  - `run_id`: Run ID to check
- **Returns:** Running status and PID or exit code

### Evaluation & Analysis

#### `list_checkpoints`
List available checkpoints for a run.
- **Parameters:**
  - `run_id`: Run ID or "@last"
- **Returns:** List of checkpoints with sizes and timestamps

#### `compare_runs`
Compare metrics across multiple runs.
- **Parameters:**
  - `run_ids`: List of run IDs to compare
  - `metrics` (optional): Specific metrics to compare
- **Returns:** Side-by-side comparison of final metrics

#### `get_best_run`
Find the best performing run for an environment.
- **Parameters:**
  - `env_id`: Environment ID
  - `metric` (optional): Metric to optimize (default: "val/roll/ep_rew/mean")
  - `minimize` (optional): Whether to minimize (default: false)
- **Returns:** Best run info and metric value

## Usage Examples

### Example 1: Discover and Launch a Training Run

```python
# Claude Code can use these tools directly:

# 1. List available environments
list_environments(filter="CartPole")
# Returns: {"count": 1, "environments": [{"env_id": "CartPole-v1", ...}]}

# 2. Check available variants
list_variants(env_id="CartPole-v1")
# Returns: {"env_id": "CartPole-v1", "variants": ["ppo", "reinforce"]}

# 3. Get config details
get_config(env_id="CartPole-v1", variant="ppo")
# Returns full config dict

# 4. Start training
start_training(
    config_id="CartPole-v1:ppo",
    max_env_steps=50000,
    wandb_mode="disabled"
)
# Returns: {"success": true, "pid": 12345, ...}
```

### Example 2: Monitor and Debug a Run

```python
# 1. Check training status
get_training_status(run_id="CartPole-v1:ppo")
# Returns: {"running": true, "pid": 12345}

# 2. Get metrics
get_run_metrics(run_id="@last", metric_names=["val/roll/ep_rew/mean"])
# Returns metrics data

# 3. Check logs if issues arise
get_run_logs(run_id="@last", lines=50)
# Returns last 50 lines of logs
```

### Example 3: Compare and Optimize

```python
# 1. List recent runs
list_runs(env_filter="CartPole", limit=5)
# Returns list of runs

# 2. Compare performance
compare_runs(
    run_ids=["run_123", "run_456"],
    metrics=["val/roll/ep_rew/mean", "total_timesteps"]
)
# Returns side-by-side comparison

# 3. Find best run
get_best_run(env_id="CartPole-v1")
# Returns best performing run

# 4. Launch improved run based on best config
start_training(
    config_id="CartPole-v1:ppo",
    max_env_steps=100000  # Increased from best run
)
```

## Agent-Guided Optimization Workflow

Here's a typical optimization cycle using the MCP server:

1. **Discovery Phase**
   - Use `list_environments` to see available envs
   - Use `list_variants` to see algorithm options
   - Use `get_config` to understand baseline configs

2. **Initial Training**
   - Use `start_training` with baseline config
   - Use `get_training_status` to monitor
   - Use `get_run_metrics` to check progress

3. **Analysis Phase**
   - Use `get_run_info` to get full run details
   - Use `get_run_logs` to debug issues
   - Use `list_checkpoints` to verify saves

4. **Comparison Phase**
   - Use `list_runs` to find related experiments
   - Use `compare_runs` to analyze differences
   - Use `get_best_run` to identify top performer

5. **Iteration Phase**
   - Modify config based on insights
   - Use `start_training` with new parameters
   - Repeat cycle

## Architecture

The MCP server (`mcp_server.py`) provides a stateless interface to gymnasium-solver's training infrastructure:

- **Environment Discovery**: Scans `config/environments/*.yaml` for available configs
- **Run Registry**: Reads `runs/runs.json` for run metadata
- **Metrics Access**: Parses `runs/<id>/metrics.csv` for performance data
- **Process Management**: Tracks training processes in `_running_processes` dict
- **Config Loading**: Uses `utils.config.load_config()` to parse YAML configs
- **Run Access**: Uses `utils.run.Run` to access run artifacts

## Limitations

1. **Process Tracking**: Only processes started via `start_training` are tracked. External training runs won't be monitored for status.

2. **Single Machine**: The server operates on the local filesystem. Distributed training requires separate infrastructure.

3. **No Video Access**: Videos are stored in checkpoints but not exposed via MCP tools (use `run_play.py` or `run_inspect.py` instead).

4. **Synchronous Operations**: Tools block until complete. Long-running metrics parsing may be slow for large runs.

## Future Enhancements

Potential additions:
- `evaluate_checkpoint`: Run evaluation episodes on a checkpoint
- `clone_config`: Create a new config based on an existing one with modifications
- `archive_run`: Move a run to an archive directory
- `export_metrics`: Export metrics in various formats (CSV, JSON, Parquet)
- `get_hyperparameter_importance`: Analyze which hyperparams matter most
- `suggest_next_config`: Bayesian optimization suggestions

## Troubleshooting

### MCP server not showing up in Claude Code
- Ensure `mcp` is installed: `uv pip install mcp`
- Check Claude Code MCP settings include the server config
- Restart Claude Code after adding the config
- Check for errors in Claude Code's MCP logs

### Training fails to start
- Verify the config_id format is "env:variant"
- Check that the environment config file exists in `config/environments/`
- Ensure Python environment has all dependencies installed
- Review logs using `get_run_logs`

### Metrics not appearing
- Wait for at least one epoch to complete
- Check that `metrics.csv` exists in the run directory
- Verify the run isn't still in warmup phase
- Use `get_run_info` to check run status

## Contributing

To add new MCP tools:

1. Add tool definition in `handle_list_tools()`
2. Add tool execution case in `handle_call_tool()`
3. Implement the async function with type hints
4. Update this documentation with usage examples
5. Test with Claude Code

## References

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Claude Code MCP Documentation](https://docs.anthropic.com/claude/docs/mcp)
- [Gymnasium-Solver Documentation](./README.md)
