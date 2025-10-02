#!/usr/bin/env python3
"""MCP server for gymnasium-solver training run management.

Provides tools for Claude Code to interact with training sessions and inspect runs.
Enables agent-guided optimization cycles: launch runs, debug, tweak, relaunch.

Usage:
    python mcp_server.py
"""

import asyncio
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Import project utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.run import Run, list_run_ids, RUNS_DIR

# Track running training processes
_running_processes: Dict[str, subprocess.Popen] = {}

server = Server("gymnasium-solver")


# ============================================================================
# Environment & Config Discovery Tools
# ============================================================================

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available MCP tools."""
    return [
        types.Tool(
            name="list_environments",
            description="List all available environment configurations",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter string to match environment names"
                    }
                },
            },
        ),
        types.Tool(
            name="list_variants",
            description="List available algorithm variants for a specific environment",
            inputSchema={
                "type": "object",
                "properties": {
                    "env_id": {
                        "type": "string",
                        "description": "Environment ID (e.g., 'CartPole-v1')"
                    }
                },
                "required": ["env_id"],
            },
        ),
        types.Tool(
            name="get_config",
            description="Get full configuration for an environment:variant combination",
            inputSchema={
                "type": "object",
                "properties": {
                    "env_id": {
                        "type": "string",
                        "description": "Environment ID (e.g., 'CartPole-v1')"
                    },
                    "variant": {
                        "type": "string",
                        "description": "Algorithm variant (e.g., 'ppo', 'reinforce')"
                    }
                },
                "required": ["env_id", "variant"],
            },
        ),
        types.Tool(
            name="list_runs",
            description="List training runs with optional filtering by environment or status",
            inputSchema={
                "type": "object",
                "properties": {
                    "env_filter": {
                        "type": "string",
                        "description": "Filter by environment ID"
                    },
                    "algo_filter": {
                        "type": "string",
                        "description": "Filter by algorithm (ppo, reinforce, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of runs to return (default: 20)"
                    }
                },
            },
        ),
        types.Tool(
            name="get_run_info",
            description="Get detailed information about a specific training run",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="get_run_metrics",
            description="Get metrics data from a training run",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "metric_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific metrics to retrieve"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Limit number of rows returned (default: all)"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="get_run_logs",
            description="Get log output from a training run for debugging",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to return from end of log (default: 100)"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="start_training",
            description="Start a new training run with specified config and overrides",
            inputSchema={
                "type": "object",
                "properties": {
                    "config_id": {
                        "type": "string",
                        "description": "Config in 'env:variant' format (e.g., 'CartPole-v1:ppo')"
                    },
                    "max_env_steps": {
                        "type": "integer",
                        "description": "Override max environment steps"
                    },
                    "quiet": {
                        "type": "boolean",
                        "description": "Run in quiet mode (default: true)"
                    },
                    "wandb_mode": {
                        "type": "string",
                        "description": "W&B mode: 'online', 'offline', or 'disabled' (default: 'disabled')"
                    }
                },
                "required": ["config_id"],
            },
        ),
        types.Tool(
            name="stop_training",
            description="Stop a running training process",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID of the training to stop"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="get_training_status",
            description="Check if a training process is still running",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to check status"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="list_checkpoints",
            description="List available checkpoints for a training run",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="compare_runs",
            description="Compare key metrics across multiple training runs",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of run IDs to compare"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to compare (default: key metrics)"
                    }
                },
                "required": ["run_ids"],
            },
        ),
        types.Tool(
            name="get_best_run",
            description="Find the best performing run for an environment based on a metric",
            inputSchema={
                "type": "object",
                "properties": {
                    "env_id": {
                        "type": "string",
                        "description": "Environment ID to search"
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric to optimize (default: 'val/roll/ep_rew/mean')"
                    },
                    "minimize": {
                        "type": "boolean",
                        "description": "Whether to minimize metric (default: false)"
                    }
                },
                "required": ["env_id"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests."""

    args = arguments or {}

    try:
        if name == "list_environments":
            result = await list_environments(args.get("filter"))
        elif name == "list_variants":
            result = await list_variants(args["env_id"])
        elif name == "get_config":
            result = await get_config(args["env_id"], args["variant"])
        elif name == "list_runs":
            result = await list_runs(
                args.get("env_filter"),
                args.get("algo_filter"),
                args.get("limit", 20)
            )
        elif name == "get_run_info":
            result = await get_run_info(args["run_id"])
        elif name == "get_run_metrics":
            result = await get_run_metrics(
                args["run_id"],
                args.get("metric_names"),
                args.get("limit")
            )
        elif name == "get_run_logs":
            result = await get_run_logs(
                args["run_id"],
                args.get("lines", 100)
            )
        elif name == "start_training":
            result = await start_training(
                args["config_id"],
                args.get("max_env_steps"),
                args.get("quiet", True),
                args.get("wandb_mode", "disabled")
            )
        elif name == "stop_training":
            result = await stop_training(args["run_id"])
        elif name == "get_training_status":
            result = await get_training_status(args["run_id"])
        elif name == "list_checkpoints":
            result = await list_checkpoints(args["run_id"])
        elif name == "compare_runs":
            result = await compare_runs(
                args["run_ids"],
                args.get("metrics")
            )
        elif name == "get_best_run":
            result = await get_best_run(
                args["env_id"],
                args.get("metric", "val/roll/ep_rew/mean"),
                args.get("minimize", False)
            )
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        return [types.TextContent(type="text", text=error_msg)]


# ============================================================================
# Tool Implementation Functions
# ============================================================================

async def list_environments(filter_str: Optional[str] = None) -> Dict[str, Any]:
    """List all available environment configurations."""
    config_dir = Path("config/environments")
    env_files = sorted(config_dir.glob("*.yaml"))

    envs = []
    for env_file in env_files:
        env_id = env_file.stem
        if filter_str and filter_str.lower() not in env_id.lower():
            continue
        envs.append({
            "env_id": env_id,
            "config_file": str(env_file)
        })

    return {
        "count": len(envs),
        "environments": envs
    }


async def list_variants(env_id: str) -> Dict[str, Any]:
    """List available algorithm variants for an environment."""
    from ruamel.yaml import YAML

    config_file = Path(f"config/environments/{env_id}.yaml")
    if not config_file.exists():
        raise FileNotFoundError(f"No config file for environment: {env_id}")

    yaml = YAML()
    with open(config_file) as f:
        data = yaml.load(f)

    # Find variant keys (top-level keys that are algorithm names)
    variants = []
    known_algos = {"ppo", "reinforce", "dqn", "a2c", "sac"}

    for key in data.keys():
        if isinstance(key, str) and key.lower() in known_algos:
            variants.append(key)

    return {
        "env_id": env_id,
        "variants": sorted(variants)
    }


async def get_config(env_id: str, variant: str) -> Dict[str, Any]:
    """Get full configuration for env:variant."""
    config = load_config(env_id, variant)

    # Convert config to dict
    from dataclasses import asdict
    config_dict = asdict(config)

    # Convert enum values to strings
    for key, value in config_dict.items():
        if hasattr(value, 'value'):
            config_dict[key] = value.value

    return {
        "config_id": f"{env_id}:{variant}",
        "config": config_dict
    }


async def list_runs(
    env_filter: Optional[str] = None,
    algo_filter: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """List training runs with filtering."""
    from utils.run import _read_registry

    registry = _read_registry()

    # Apply filters
    filtered = registry
    if env_filter:
        filtered = [r for r in filtered if env_filter in r.get("env_id", "")]
    if algo_filter:
        filtered = [r for r in filtered if algo_filter in r.get("algo", "")]

    # Limit results
    filtered = filtered[:limit]

    return {
        "count": len(filtered),
        "total": len(registry),
        "runs": filtered
    }


async def get_run_info(run_id: str) -> Dict[str, Any]:
    """Get detailed information about a training run."""
    run = Run.from_id(run_id)
    config = run.load_config()

    # Get checkpoint info
    checkpoints_dir = Path(run.run_dir) / "checkpoints"
    checkpoints = []
    if checkpoints_dir.exists():
        for ckpt in sorted(checkpoints_dir.glob("epoch=*.ckpt")):
            checkpoints.append(ckpt.name)

    # Get metrics summary if available
    metrics_file = Path(run.run_dir) / "metrics.csv"
    metrics_summary = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                # Include key metrics
                for key in ["train/roll/ep_rew/mean", "val/roll/ep_rew/mean",
                           "train/roll/ep_len/mean", "total_timesteps"]:
                    if key in last_row:
                        metrics_summary[key] = last_row[key]

    from dataclasses import asdict
    return {
        "run_id": run.run_id,
        "run_dir": run.run_dir,
        "config": asdict(config),
        "checkpoints": checkpoints,
        "metrics_summary": metrics_summary
    }


async def get_run_metrics(
    run_id: str,
    metric_names: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Get metrics data from a training run."""
    run = Run.from_id(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Apply limit
    if limit:
        rows = rows[-limit:]

    # Filter columns if specified
    if metric_names:
        filtered_rows = []
        for row in rows:
            filtered_row = {k: v for k, v in row.items() if k in metric_names or k == "epoch"}
            filtered_rows.append(filtered_row)
        rows = filtered_rows

    return {
        "run_id": run_id,
        "rows": len(rows),
        "metrics": rows
    }


async def get_run_logs(run_id: str, lines: int = 100) -> Dict[str, Any]:
    """Get log output from a training run."""
    run = Run.from_id(run_id)
    log_file = Path(run.run_dir) / "run.log"

    if not log_file.exists():
        return {"error": f"No log file found for run {run_id}"}

    # Read last N lines
    with open(log_file) as f:
        all_lines = f.readlines()
        last_lines = all_lines[-lines:]

    return {
        "run_id": run_id,
        "total_lines": len(all_lines),
        "returned_lines": len(last_lines),
        "log": "".join(last_lines)
    }


async def start_training(
    config_id: str,
    max_env_steps: Optional[int] = None,
    quiet: bool = True,
    wandb_mode: str = "disabled"
) -> Dict[str, Any]:
    """Start a new training run."""

    # Build command
    cmd = ["python", "train.py", config_id]

    if max_env_steps:
        cmd.extend(["--max-env-steps", str(max_env_steps)])

    # Set environment variables
    env = dict(os.environ)
    env["WANDB_MODE"] = wandb_mode
    if quiet:
        env["VIBES_QUIET"] = "1"

    # Start process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )

    # Wait a moment to check if it started successfully
    await asyncio.sleep(2)

    if process.poll() is not None:
        # Process already exited
        stdout, stderr = process.communicate()
        return {
            "success": False,
            "error": "Training process exited immediately",
            "stdout": stdout,
            "stderr": stderr
        }

    # Track process
    _running_processes[config_id] = process

    return {
        "success": True,
        "config_id": config_id,
        "pid": process.pid,
        "message": "Training started. Use get_training_status to monitor progress."
    }


async def stop_training(run_id: str) -> Dict[str, Any]:
    """Stop a running training process."""
    if run_id not in _running_processes:
        return {"error": f"No running process found for {run_id}"}

    process = _running_processes[run_id]
    process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()

    del _running_processes[run_id]

    return {
        "success": True,
        "run_id": run_id,
        "message": "Training process stopped"
    }


async def get_training_status(run_id: str) -> Dict[str, Any]:
    """Check if a training process is still running."""
    if run_id not in _running_processes:
        return {
            "run_id": run_id,
            "running": False,
            "message": "No tracked process found"
        }

    process = _running_processes[run_id]
    poll_result = process.poll()

    if poll_result is None:
        return {
            "run_id": run_id,
            "running": True,
            "pid": process.pid
        }
    else:
        return {
            "run_id": run_id,
            "running": False,
            "exit_code": poll_result
        }


async def list_checkpoints(run_id: str) -> Dict[str, Any]:
    """List available checkpoints for a run."""
    run = Run.from_id(run_id)
    checkpoints_dir = Path(run.run_dir) / "checkpoints"

    if not checkpoints_dir.exists():
        return {"error": f"No checkpoints directory for run {run_id}"}

    checkpoints = []
    for ckpt in sorted(checkpoints_dir.glob("epoch=*.ckpt")):
        checkpoints.append({
            "name": ckpt.name,
            "size_mb": ckpt.stat().st_size / 1024 / 1024,
            "modified": ckpt.stat().st_mtime
        })

    # Check for best/last symlinks
    best_link = checkpoints_dir / "best.ckpt"
    last_link = checkpoints_dir / "last.ckpt"

    return {
        "run_id": run_id,
        "checkpoints": checkpoints,
        "has_best": best_link.exists(),
        "has_last": last_link.exists()
    }


async def compare_runs(
    run_ids: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compare metrics across multiple runs."""

    # Default key metrics
    if not metrics:
        metrics = [
            "val/roll/ep_rew/mean",
            "val/roll/ep_rew/best",
            "train/roll/ep_rew/mean",
            "total_timesteps"
        ]

    comparison = {}
    for run_id in run_ids:
        try:
            run = Run.from_id(run_id)
            metrics_file = Path(run.run_dir) / "metrics.csv"

            if not metrics_file.exists():
                comparison[run_id] = {"error": "No metrics file"}
                continue

            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                comparison[run_id] = {"error": "No data"}
                continue

            last_row = rows[-1]
            run_metrics = {}
            for metric in metrics:
                if metric in last_row:
                    run_metrics[metric] = last_row[metric]

            comparison[run_id] = run_metrics

        except Exception as e:
            comparison[run_id] = {"error": str(e)}

    return {
        "runs_compared": len(run_ids),
        "metrics": metrics,
        "comparison": comparison
    }


async def get_best_run(
    env_id: str,
    metric: str = "val/roll/ep_rew/mean",
    minimize: bool = False
) -> Dict[str, Any]:
    """Find the best performing run for an environment."""
    from utils.run import _read_registry

    registry = _read_registry()

    # Filter by environment
    env_runs = [r for r in registry if r.get("env_id") == env_id]

    if not env_runs:
        return {"error": f"No runs found for environment {env_id}"}

    # Evaluate each run
    best_run = None
    best_value = float('inf') if minimize else float('-inf')

    for run_entry in env_runs:
        run_id = run_entry["run_id"]
        try:
            metrics_file = RUNS_DIR / run_id / "metrics.csv"

            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows or metric not in rows[-1]:
                continue

            value = float(rows[-1][metric])

            if minimize:
                if value < best_value:
                    best_value = value
                    best_run = run_entry
            else:
                if value > best_value:
                    best_value = value
                    best_run = run_entry

        except Exception:
            continue

    if best_run is None:
        return {"error": f"No valid runs found for {env_id} with metric {metric}"}

    return {
        "env_id": env_id,
        "metric": metric,
        "best_value": best_value,
        "best_run": best_run
    }


# ============================================================================
# Server Main
# ============================================================================

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="gymnasium-solver",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
