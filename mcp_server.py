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
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Change to script directory to ensure correct working directory
SCRIPT_DIR = Path(__file__).parent.absolute()
os.chdir(SCRIPT_DIR)

# Import project utilities
sys.path.insert(0, str(SCRIPT_DIR))
from utils.config import load_config
from utils.run import Run, RUNS_DIR

# Track running training processes
_running_processes: Dict[str, subprocess.Popen] = {}

server = Server("gymnasium-solver")


# ============================================================================
# Helper Functions
# ============================================================================

def fuzzy_match_metrics(
    requested_metrics: List[str],
    available_metrics: List[str],
    cutoff: float = 0.6
) -> Tuple[List[str], List[Tuple[str, List[str]]]]:
    """Fuzzy match requested metrics to available metrics.

    Returns:
        (matched_metrics, suggestions) where suggestions is [(bad_metric, [similar_metrics])]
    """
    matched = []
    suggestions = []

    for metric in requested_metrics:
        if metric in available_metrics:
            matched.append(metric)
        else:
            # Try fuzzy matching
            similar = get_close_matches(metric, available_metrics, n=5, cutoff=cutoff)
            if similar:
                suggestions.append((metric, similar))

    return matched, suggestions


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
                    "overrides": {
                        "type": "object",
                        "description": "Dict of config field overrides (e.g., {'policy_lr': 0.001, 'batch_size': 64})"
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
            description="Check if a training process is still running, with optional detailed progress info including progress %, metrics, and recent logs",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to check status"
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include detailed progress, metrics, and logs (default: true)"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="wait_for_training_completion",
            description="Wait for a training run to complete and return final results. Blocks until training finishes or timeout is reached.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds to wait (default: 3600 = 1 hour)"
                    },
                    "poll_interval": {
                        "type": "integer",
                        "description": "Seconds between status checks (default: 5)"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="stream_training_logs",
            description="Get training logs with optional filtering by regex pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "follow": {
                        "type": "boolean",
                        "description": "If true, returns most recent lines (default: false)"
                    },
                    "filter_pattern": {
                        "type": "string",
                        "description": "Optional regex pattern to filter log lines"
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines to return (default: 50)"
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
        types.Tool(
            name="plot_run_metric",
            description="Generate a plot for one or more metrics from a training run",
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
                        "description": "List of metric names to plot"
                    },
                    "x_axis": {
                        "type": "string",
                        "description": "X-axis column (default: 'epoch')"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional plot title"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Plot width in pixels (default: 800)"
                    },
                    "height": {
                        "type": "integer",
                        "description": "Plot height in pixels (default: 600)"
                    }
                },
                "required": ["run_id", "metric_names"],
            },
        ),
        types.Tool(
            name="list_available_metrics",
            description="List all metric column names available in a run's metrics.csv",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "filter": {
                        "type": "string",
                        "description": "Optional filter to match metric names (e.g., 'loss', 'entropy')"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="get_metric_alerts",
            description="Extract metric alerts/warnings from training run logs",
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
            name="get_metrics_range",
            description="Get metrics for a specific epoch range (useful for comparing early vs late training)",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "start_epoch": {
                        "type": "integer",
                        "description": "Starting epoch (inclusive)"
                    },
                    "end_epoch": {
                        "type": "integer",
                        "description": "Ending epoch (inclusive)"
                    },
                    "metric_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of specific metrics to retrieve"
                    }
                },
                "required": ["run_id", "start_epoch", "end_epoch"],
            },
        ),
        types.Tool(
            name="get_metrics_summary",
            description="Get statistical summary of a metric across training (min, max, mean, std, trend)",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "metric_name": {
                        "type": "string",
                        "description": "Metric to summarize (e.g., 'train/opt/loss/policy')"
                    }
                },
                "required": ["run_id", "metric_name"],
            },
        ),
        types.Tool(
            name="get_training_progress",
            description="Get quick snapshot of training progress and health",
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
            name="correlate_metrics",
            description="Calculate correlation between pairs of metrics to identify relationships",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "metric_pairs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "description": "List of metric pairs to correlate, e.g., [['metric1', 'metric2'], ...]"
                    }
                },
                "required": ["run_id", "metric_pairs"],
            },
        ),
        types.Tool(
            name="get_metric_trend",
            description="Analyze trend direction and magnitude for a metric over a window",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric to analyze trend for"
                    },
                    "window": {
                        "type": "string",
                        "description": "Window to analyze: 'all', 'last_N_epochs', 'first_N_epochs', e.g., 'last_20_epochs'"
                    }
                },
                "required": ["run_id", "metric"],
            },
        ),
        types.Tool(
            name="compare_to_baseline",
            description="Compare a run's performance to a baseline (best run or specific run)",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to compare"
                    },
                    "baseline": {
                        "type": "string",
                        "description": "Baseline to compare against: 'best_for_env' or a specific run_id"
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric to compare (default: 'train/roll/ep_rew/mean')"
                    }
                },
                "required": ["run_id", "baseline"],
            },
        ),
        types.Tool(
            name="health_check",
            description="Quick health check identifying top anomalies ranked by severity",
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
            name="plot_compare_runs",
            description="Plot the same metric across multiple runs for comparison",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of run IDs to compare"
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric to plot across all runs"
                    },
                    "x_axis": {
                        "type": "string",
                        "description": "X-axis column (default: 'epoch')"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional plot title"
                    },
                    "width": {
                        "type": "integer",
                        "description": "Plot width in pixels (default: 800)"
                    },
                    "height": {
                        "type": "integer",
                        "description": "Plot height in pixels (default: 600)"
                    }
                },
                "required": ["run_ids", "metric"],
            },
        ),
        types.Tool(
            name="get_hyperparam_history",
            description="Track hyperparameter values across training (for scheduled hyperparams)",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of hyperparameters to track (e.g., ['policy_lr', 'ent_coef'])"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="comprehensive_diagnostic",
            description="Single comprehensive diagnostic call that returns all essential run information for analysis. This is the PREFERRED tool for run debugging - it consolidates status, config, progress, key metrics, health checks, and trend analysis into one efficient call. Use this instead of calling multiple individual tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID or '@last' for most recent run"
                    },
                    "include_recent_metrics": {
                        "type": "boolean",
                        "description": "Include last 20 epochs of key metrics for trend analysis (default: true)"
                    },
                    "include_full_config": {
                        "type": "boolean",
                        "description": "Include full config details vs summary (default: false)"
                    }
                },
                "required": ["run_id"],
            },
        ),
        types.Tool(
            name="run_play",
            description="Play a trained policy or test an environment with random/user policy. Supports visualization options like CNN filters and preprocessing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to play (use '@last' for most recent). Mutually exclusive with config_id."
                    },
                    "config_id": {
                        "type": "string",
                        "description": "Config ID in 'env:variant' format (e.g., 'CartPole-v1:ppo') for testing with random/user policy. Mutually exclusive with run_id."
                    },
                    "episodes": {
                        "type": "integer",
                        "description": "Number of episodes to play (default: 1)"
                    },
                    "deterministic": {
                        "type": "boolean",
                        "description": "Use deterministic actions (default: false)"
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Do not render the environment (default: true for API calls)"
                    },
                    "mode": {
                        "type": "string",
                        "description": "Action mode: 'trained', 'random', or 'user' (keyboard input)"
                    },
                    "seed": {
                        "type": "string",
                        "description": "Random seed for environment (int, 'train', 'val', 'test', or None)"
                    },
                    "env_kwargs": {
                        "type": "object",
                        "description": "Override env_kwargs fields (e.g., {'state': 'Level2-1'})"
                    },
                    "show_preprocessing": {
                        "type": "boolean",
                        "description": "Show preprocessed observations (default: false)"
                    },
                    "show_cnn_filters": {
                        "type": "boolean",
                        "description": "Show CNN filters and activations (default: false)"
                    }
                },
            },
        ),
        types.Tool(
            name="run_inspect",
            description="Launch Gradio inspection UI for a training run. Runs in background as a web server.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to inspect (default: '@last')"
                    },
                    "port": {
                        "type": "integer",
                        "description": "Port for Gradio server (default: 7860)"
                    },
                    "host": {
                        "type": "string",
                        "description": "Host for Gradio server (default: 'localhost')"
                    },
                    "share": {
                        "type": "boolean",
                        "description": "Enable Gradio share link (default: false)"
                    },
                    "seed": {
                        "type": "string",
                        "description": "Random seed for environment (int, 'train', 'val', 'test', or None)"
                    },
                    "env_kwargs": {
                        "type": "object",
                        "description": "Override env_kwargs fields (e.g., {'state': 'Level2-1'})"
                    }
                },
            },
        ),
        types.Tool(
            name="run_publish",
            description="Publish a training run to Hugging Face Hub",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Run ID to publish (default: '@last')"
                    },
                    "repo": {
                        "type": "string",
                        "description": "Target repo id (e.g., 'user/repo'). If omitted, will be inferred."
                    },
                    "private": {
                        "type": "boolean",
                        "description": "Create repo as private (default: false)"
                    }
                },
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
                args.get("overrides"),
                args.get("quiet", True),
                args.get("wandb_mode", "disabled")
            )
        elif name == "stop_training":
            result = await stop_training(args["run_id"])
        elif name == "get_training_status":
            result = await get_training_status(
                args["run_id"],
                args.get("include_details", True)
            )
        elif name == "wait_for_training_completion":
            result = await wait_for_training_completion(
                args["run_id"],
                args.get("timeout", 3600),
                args.get("poll_interval", 5)
            )
        elif name == "stream_training_logs":
            result = await stream_training_logs(
                args["run_id"],
                args.get("follow", False),
                args.get("filter_pattern"),
                args.get("lines", 50)
            )
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
        elif name == "plot_run_metric":
            return await plot_run_metric(
                args["run_id"],
                args["metric_names"],
                args.get("x_axis", "epoch"),
                args.get("title"),
                args.get("width", 800),
                args.get("height", 600)
            )
        elif name == "list_available_metrics":
            result = await list_available_metrics(
                args["run_id"],
                args.get("filter")
            )
        elif name == "get_metric_alerts":
            result = await get_metric_alerts(args["run_id"])
        elif name == "get_metrics_range":
            result = await get_metrics_range(
                args["run_id"],
                args["start_epoch"],
                args["end_epoch"],
                args.get("metric_names")
            )
        elif name == "get_metrics_summary":
            result = await get_metrics_summary(
                args["run_id"],
                args["metric_name"]
            )
        elif name == "get_training_progress":
            result = await get_training_progress(args["run_id"])
        elif name == "correlate_metrics":
            result = await correlate_metrics(
                args["run_id"],
                args["metric_pairs"]
            )
        elif name == "get_metric_trend":
            result = await get_metric_trend(
                args["run_id"],
                args["metric"],
                args.get("window", "all")
            )
        elif name == "compare_to_baseline":
            result = await compare_to_baseline(
                args["run_id"],
                args["baseline"],
                args.get("metric", "train/roll/ep_rew/mean")
            )
        elif name == "health_check":
            result = await health_check(args["run_id"])
        elif name == "plot_compare_runs":
            return await plot_compare_runs(
                args["run_ids"],
                args["metric"],
                args.get("x_axis", "epoch"),
                args.get("title"),
                args.get("width", 800),
                args.get("height", 600)
            )
        elif name == "get_hyperparam_history":
            result = await get_hyperparam_history(
                args["run_id"],
                args.get("params")
            )
        elif name == "comprehensive_diagnostic":
            result = await comprehensive_diagnostic(
                args["run_id"],
                args.get("include_recent_metrics", True),
                args.get("include_full_config", False)
            )
        elif name == "run_play":
            result = await run_play(
                args.get("run_id"),
                args.get("config_id"),
                args.get("episodes", 1),
                args.get("deterministic", False),
                args.get("headless", True),
                args.get("mode"),
                args.get("seed"),
                args.get("env_kwargs"),
                args.get("show_preprocessing", False),
                args.get("show_cnn_filters", False)
            )
        elif name == "run_inspect":
            result = await run_inspect(
                args.get("run_id", "@last"),
                args.get("port", 7860),
                args.get("host", "localhost"),
                args.get("share", False),
                args.get("seed"),
                args.get("env_kwargs")
            )
        elif name == "run_publish":
            result = await run_publish(
                args.get("run_id", "@last"),
                args.get("repo"),
                args.get("private", False)
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
    import os
    cwd = os.getcwd()
    config_dir = Path("config/environments")
    config_dir_abs = config_dir.absolute()
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
        "environments": envs,
        "_debug": {
            "cwd": cwd,
            "config_dir": str(config_dir),
            "config_dir_abs": str(config_dir_abs),
            "exists": config_dir.exists(),
            "files_found": len(env_files)
        }
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
    run = Run.load(run_id)
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
        "run_dir": str(run.run_dir),
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
    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Apply limit
    if limit:
        rows = rows[-limit:]

    # Default to key metrics if not specified to avoid huge responses
    if not metric_names:
        metric_names = [
            "epoch",
            "train/roll/ep_rew/mean",
            "train/roll/ep_rew/best",
            "val/roll/ep_rew/mean",
            "val/roll/ep_rew/best",
            "total_timesteps"
        ]

    # Filter columns and remove null/empty values to save tokens
    filtered_rows = []
    for row in rows:
        filtered_row = {k: v for k, v in row.items()
                       if k in metric_names and v not in (None, "", "null")}
        filtered_rows.append(filtered_row)

    return {
        "run_id": run_id,
        "rows": len(filtered_rows),
        "metrics": filtered_rows
    }


async def get_run_logs(run_id: str, lines: int = 100) -> Dict[str, Any]:
    """Get log output from a training run."""
    run = Run.load(run_id)
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
    overrides: Optional[Dict[str, Any]] = None,
    quiet: bool = True,
    wandb_mode: str = "disabled"
) -> Dict[str, Any]:
    """Start a new training run."""

    # Build command
    cmd = ["python", "train.py", config_id]

    if max_env_steps:
        cmd.extend(["--max-env-steps", str(max_env_steps)])

    # Add config overrides
    if overrides:
        for key, value in overrides.items():
            cmd.extend(["--override", f"{key}={value}"])

    # Set environment variables
    env = dict(os.environ)
    env["WANDB_MODE"] = wandb_mode
    # Always set VIBES_QUIET when running via MCP to disable progress bars
    # Progress bars don't work properly when stdout is redirected
    env["VIBES_QUIET"] = "1"

    # Start process in new session to fully detach from terminal
    # Redirect stdout/stderr to a temp log file (will be in run directory anyway)
    # Using a file instead of DEVNULL to avoid potential multiprocessing issues
    log_file = open("/tmp/mcp_training.log", "a")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        env=env,
        text=True,
        start_new_session=True  # Detach from controlling terminal
    )

    # Wait a moment to check if it started successfully
    await asyncio.sleep(2)

    if process.poll() is not None:
        # Process already exited
        return {
            "success": False,
            "error": f"Training process exited immediately with code {process.returncode}",
            "message": "Check the run logs for details"
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


async def get_training_status(run_id: str, include_details: bool = True) -> Dict[str, Any]:
    """Check if a training process is still running, with optional detailed progress info.

    Args:
        run_id: Run ID to check
        include_details: If True, includes progress %, current metrics, and recent logs
    """
    import time

    # First check tracked processes
    is_running = False
    pid = None
    source = None

    if run_id in _running_processes:
        process = _running_processes[run_id]
        poll_result = process.poll()

        if poll_result is None:
            is_running = True
            pid = process.pid
            source = "tracked"
        else:
            # Process finished, remove from tracking
            del _running_processes[run_id]

    # Fall back to system-wide process check if not found in tracked
    if not is_running:
        try:
            run = Run.load(run_id)
            run_dir = run.run_dir

            # Check metrics.csv timestamp to see if recently updated
            metrics_file = Path(run_dir) / "metrics.csv"
            if metrics_file.exists():
                mtime = metrics_file.stat().st_mtime
                age = time.time() - mtime
                if age < 60:  # Updated in last minute
                    # Find any train.py process
                    result = subprocess.run(
                        ["pgrep", "-f", "python.*train.py"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                        is_running = True
                        pid = pids[0] if len(pids) == 1 else pids
                        source = "system"
        except Exception:
            pass

    # Build response
    response = {
        "run_id": run_id,
        "running": is_running
    }

    if is_running:
        response["pid"] = pid
        response["source"] = source

    if not is_running:
        response["message"] = "No running process found"
        return response

    # Add detailed information if requested and running
    if include_details and is_running:
        try:
            # Get progress from get_training_progress
            progress_info = await get_training_progress(run_id)
            if "progress" in progress_info:
                response["progress"] = progress_info["progress"]
                response["performance"] = progress_info.get("performance", {})

            # Get last 10 lines of logs
            logs_result = await get_run_logs(run_id, lines=10)
            if "log" in logs_result:
                response["recent_logs"] = logs_result["log"].strip().split('\n')[-10:]

        except Exception as e:
            response["details_error"] = str(e)

    return response


async def wait_for_training_completion(
    run_id: str,
    timeout: int = 3600,
    poll_interval: int = 5
) -> Dict[str, Any]:
    """Wait for a training run to complete and return final results.

    Args:
        run_id: Run ID to wait for (supports '@last')
        timeout: Maximum seconds to wait (default: 3600 = 1 hour)
        poll_interval: Seconds between status checks (default: 5)

    Returns:
        Dict with final status, metrics, and completion reason
    """
    import time

    start_time = time.time()
    last_status = None

    while (time.time() - start_time) < timeout:
        # Check if still running
        status = await get_training_status(run_id)
        last_status = status

        if not status.get("running", False):
            # Training completed! Get final results
            try:
                run_info = await get_run_info(run_id)

                # Get last few log lines to check completion reason
                logs_result = await get_run_logs(run_id, lines=50)
                logs = logs_result.get("log", "")

                # Parse completion reason from logs
                completion_reason = "unknown"
                if "Training completed" in logs:
                    # Extract reason from log line like: "Training completed in X seconds. Reason: ..."
                    import re
                    match = re.search(r"Reason: (.+?)(?:\.|$)", logs)
                    if match:
                        completion_reason = match.group(1)
                elif "KeyboardInterrupt" in logs or "SIGTERM" in logs:
                    completion_reason = "interrupted"
                elif "error" in logs.lower() or "exception" in logs.lower():
                    completion_reason = "error"

                return {
                    "success": True,
                    "run_id": run_id,
                    "status": "completed",
                    "completion_reason": completion_reason,
                    "elapsed_time": time.time() - start_time,
                    "final_metrics": run_info.get("metrics_summary", {}),
                    "config_summary": run_info.get("config", {})
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Training completed but failed to get results: {str(e)}",
                    "elapsed_time": time.time() - start_time
                }

        # Still running, wait before next check
        await asyncio.sleep(poll_interval)

    # Timeout reached
    return {
        "success": False,
        "status": "timeout",
        "error": f"Training did not complete within {timeout} seconds",
        "elapsed_time": time.time() - start_time,
        "last_status": last_status
    }


async def stream_training_logs(
    run_id: str,
    follow: bool = False,
    filter_pattern: Optional[str] = None,
    lines: int = 50
) -> Dict[str, Any]:
    """Get training logs with optional filtering.

    Args:
        run_id: Run ID (supports '@last')
        follow: If True, returns most recent logs (last N lines)
        filter_pattern: Optional regex pattern to filter log lines
        lines: Number of lines to return (default: 50)

    Returns:
        Dict with filtered log content
    """
    # Get logs using existing get_run_logs
    logs_result = await get_run_logs(run_id, lines=lines)

    if "error" in logs_result:
        return logs_result

    log_content = logs_result.get("log", "")
    log_lines = log_content.strip().split('\n')

    # Apply filter if provided
    if filter_pattern:
        import re
        try:
            pattern = re.compile(filter_pattern)
            log_lines = [line for line in log_lines if pattern.search(line)]
        except re.error as e:
            return {
                "error": f"Invalid regex pattern: {str(e)}",
                "pattern": filter_pattern
            }

    # If follow mode, take last N lines after filtering
    if follow and len(log_lines) > lines:
        log_lines = log_lines[-lines:]

    return {
        "run_id": run_id,
        "log": '\n'.join(log_lines),
        "lines_returned": len(log_lines),
        "total_lines": logs_result.get("total_lines"),
        "filtered": filter_pattern is not None,
        "filter_pattern": filter_pattern
    }


async def list_checkpoints(run_id: str) -> Dict[str, Any]:
    """List available checkpoints for a run."""
    run = Run.load(run_id)
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
            run = Run.load(run_id)
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


async def plot_run_metric(
    run_id: str,
    metric_names: List[str],
    x_axis: str = "epoch",
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600
) -> list[types.ImageContent]:
    """Generate a plot for metrics from a training run."""
    import base64
    import io
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Load metrics
    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"No metrics file found for run {run_id}"})
        )]

    # Read CSV data
    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "No data in metrics file"})
        )]

    # Check if x_axis column exists
    if x_axis not in rows[0]:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": f"X-axis column '{x_axis}' not found in metrics"})
        )]

    # Check if all metrics exist (with fuzzy matching)
    available_metrics = list(rows[0].keys())
    matched_metrics, suggestions = fuzzy_match_metrics(metric_names, available_metrics)

    if suggestions:
        error_msg = {"error": "Some metrics not found", "suggestions": {}}
        for bad_metric, similar in suggestions:
            error_msg["suggestions"][bad_metric] = similar
        return [types.TextContent(
            type="text",
            text=json.dumps(error_msg, indent=2)
        )]

    # Use matched metrics (may be a subset if some failed fuzzy match)
    metric_names = matched_metrics
    if not metric_names:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "No valid metrics to plot"})
        )]

    # Extract data
    x_data = []
    y_data = {metric: [] for metric in metric_names}

    for row in rows:
        if x_axis in row and row[x_axis]:
            x_val = float(row[x_axis])
            x_data.append(x_val)

            for metric in metric_names:
                if metric in row and row[metric]:
                    try:
                        y_data[metric].append(float(row[metric]))
                    except (ValueError, TypeError):
                        y_data[metric].append(None)
                else:
                    y_data[metric].append(None)

    # Create plot
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

    # Plot each metric
    for metric in metric_names:
        # Filter out None values
        valid_indices = [i for i, v in enumerate(y_data[metric]) if v is not None]
        valid_x = [x_data[i] for i in valid_indices]
        valid_y = [y_data[metric][i] for i in valid_indices]

        if valid_x and valid_y:
            ax.plot(valid_x, valid_y, label=metric, marker='o', markersize=3)

    ax.set_xlabel(x_axis)
    ax.set_ylabel('Value')
    ax.set_title(title or f"Metrics for run {run_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Encode as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return [types.ImageContent(
        type="image",
        data=img_base64,
        mimeType="image/png"
    )]


async def list_available_metrics(
    run_id: str,
    filter_str: Optional[str] = None
) -> Dict[str, Any]:
    """List all metric columns available in metrics.csv."""
    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        all_columns = reader.fieldnames

    if not all_columns:
        return {"error": "No columns found in metrics file"}

    # Apply filter
    if filter_str:
        filtered = [col for col in all_columns if filter_str.lower() in col.lower()]
    else:
        filtered = all_columns

    # Group by namespace
    grouped = {}
    for col in filtered:
        if "/" in col:
            namespace = col.split("/")[0]
            if namespace not in grouped:
                grouped[namespace] = []
            grouped[namespace].append(col)
        else:
            if "other" not in grouped:
                grouped["other"] = []
            grouped["other"].append(col)

    return {
        "run_id": run_id,
        "total_metrics": len(all_columns),
        "filtered_metrics": len(filtered),
        "metrics": filtered,
        "grouped": grouped
    }


async def get_metric_alerts(run_id: str) -> Dict[str, Any]:
    """Extract and parse metric alerts from training logs."""
    run = Run.load(run_id)
    log_file = Path(run.run_dir) / "run.log"

    if not log_file.exists():
        return {"error": f"No log file found for run {run_id}"}

    # Read log file and find alerts section
    with open(log_file) as f:
        log_content = f.read()

    # Look for METRIC ALERTS section
    alerts_start = log_content.find("METRIC ALERTS")
    if alerts_start == -1:
        return {
            "run_id": run_id,
            "alerts": [],
            "message": "No metric alerts section found in logs"
        }

    # Extract alerts section
    alerts_section = log_content[alerts_start:]
    alerts_end = alerts_section.find("\n" + "═" * 48 + "\n")
    if alerts_end != -1:
        alerts_section = alerts_section[:alerts_end]

    # Parse individual alerts
    alerts = []
    lines = alerts_section.split("\n")
    current_alert = None

    for line in lines:
        line = line.strip()
        if not line or "METRIC ALERTS" in line or "═" in line:
            continue

        # New alert starts with "- `"
        if line.startswith("- `"):
            if current_alert:
                alerts.append(current_alert)

            # Parse alert header
            # Format: - `metric_name/alert_type` triggered in `X/Y (Z%)` epochs of training:
            parts = line.split("` triggered in `")
            if len(parts) == 2:
                metric_alert = parts[0].replace("- `", "")
                frequency_part = parts[1].split("` epochs")[0]

                current_alert = {
                    "metric_alert": metric_alert,
                    "frequency": frequency_part,
                    "message": "",
                    "tip": ""
                }
        elif current_alert:
            # Parse message and tip
            if "- message:" in line:
                current_alert["message"] = line.replace("- message:", "").strip()
            elif "- tip:" in line:
                current_alert["tip"] = line.replace("- tip:", "").strip()

    # Add last alert
    if current_alert:
        alerts.append(current_alert)

    return {
        "run_id": run_id,
        "alert_count": len(alerts),
        "alerts": alerts
    }


async def get_metrics_range(
    run_id: str,
    start_epoch: int,
    end_epoch: int,
    metric_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Get metrics for a specific epoch range."""
    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Filter by epoch range
    filtered_rows = []
    for row in all_rows:
        if "epoch" in row and row["epoch"]:
            try:
                epoch = float(row["epoch"])
                if start_epoch <= epoch <= end_epoch:
                    filtered_rows.append(row)
            except (ValueError, TypeError):
                continue

    if not filtered_rows:
        return {
            "error": f"No data found for epoch range {start_epoch}-{end_epoch}"
        }

    # Default to key metrics if not specified to avoid huge responses
    if not metric_names:
        metric_names = [
            "epoch",
            "train/roll/ep_rew/mean",
            "train/roll/ep_rew/best",
            "val/roll/ep_rew/mean",
            "val/roll/ep_rew/best",
            "total_timesteps"
        ]

    # Filter columns and remove null/empty values to save tokens
    result_rows = []
    for row in filtered_rows:
        filtered_row = {k: v for k, v in row.items()
                      if k in metric_names and v not in (None, "", "null")}
        result_rows.append(filtered_row)

    return {
        "run_id": run_id,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "rows": len(result_rows),
        "metrics": result_rows
    }


async def get_metrics_summary(
    run_id: str,
    metric_name: str
) -> Dict[str, Any]:
    """Get statistical summary of a metric."""
    import statistics

    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if metric_name not in rows[0]:
        return {"error": f"Metric '{metric_name}' not found in metrics file"}

    # Extract values
    values = []
    epochs = []
    for row in rows:
        if metric_name in row and row[metric_name]:
            try:
                val = float(row[metric_name])
                values.append(val)
                if "epoch" in row and row["epoch"]:
                    epochs.append(float(row["epoch"]))
            except (ValueError, TypeError):
                continue

    if not values:
        return {"error": f"No valid data for metric '{metric_name}'"}

    # Calculate statistics
    min_val = min(values)
    max_val = max(values)
    mean_val = statistics.mean(values)

    if len(values) > 1:
        std_val = statistics.stdev(values)

        # Simple trend: compare first 10% to last 10%
        window = max(1, len(values) // 10)
        early_mean = statistics.mean(values[:window])
        late_mean = statistics.mean(values[-window:])
        trend = "improving" if late_mean > early_mean else "declining" if late_mean < early_mean else "stable"
        trend_delta = late_mean - early_mean
    else:
        std_val = 0.0
        trend = "insufficient_data"
        trend_delta = 0.0

    return {
        "run_id": run_id,
        "metric": metric_name,
        "count": len(values),
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "first": values[0],
        "last": values[-1],
        "trend": trend,
        "trend_delta": trend_delta,
        "epoch_range": [epochs[0], epochs[-1]] if epochs else None
    }


async def get_training_progress(run_id: str) -> Dict[str, Any]:
    """Get snapshot of training progress and health."""
    run = Run.load(run_id)
    config = run.load_config()
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "No metrics data available"}

    last_row = rows[-1]

    # Calculate progress
    total_timesteps = float(last_row.get("total_timesteps", 0))
    max_env_steps = getattr(config, "max_env_steps", None)

    if max_env_steps:
        progress_pct = (total_timesteps / max_env_steps) * 100
    else:
        progress_pct = None

    # Get current performance
    train_reward = last_row.get("train/roll/ep_rew/mean", "")
    val_reward = last_row.get("val/roll/ep_rew/mean", "")
    best_reward = last_row.get("train/roll/ep_rew/best", "")

    # Check if solved
    reward_threshold = getattr(config, "reward_threshold", None)
    is_solved = False
    if reward_threshold and train_reward:
        try:
            is_solved = float(train_reward) >= reward_threshold
        except (ValueError, TypeError):
            pass

    return {
        "run_id": run_id,
        "progress": {
            "total_timesteps": total_timesteps,
            "max_env_steps": max_env_steps,
            "progress_pct": progress_pct,
            "current_epoch": last_row.get("epoch", "")
        },
        "performance": {
            "train_reward_mean": train_reward,
            "val_reward_mean": val_reward,
            "best_reward": best_reward,
            "reward_threshold": reward_threshold,
            "is_solved": is_solved
        },
        "config_summary": {
            "env_id": config.env_id,
            "n_envs": config.n_envs,
            "policy_lr": config.policy_lr,
            "batch_size": config.batch_size
        }
    }


async def correlate_metrics(
    run_id: str,
    metric_pairs: List[List[str]]
) -> Dict[str, Any]:
    """Calculate correlation between pairs of metrics."""
    import statistics

    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "No metrics data available"}

    correlations = []

    for pair in metric_pairs:
        if len(pair) != 2:
            correlations.append({
                "metrics": pair,
                "error": "Pair must contain exactly 2 metrics"
            })
            continue

        metric1, metric2 = pair

        # Check metrics exist
        if metric1 not in rows[0] or metric2 not in rows[0]:
            missing = []
            if metric1 not in rows[0]:
                missing.append(metric1)
            if metric2 not in rows[0]:
                missing.append(metric2)
            correlations.append({
                "metrics": pair,
                "error": f"Metrics not found: {missing}"
            })
            continue

        # Extract paired values
        values1 = []
        values2 = []
        for row in rows:
            if metric1 in row and metric2 in row and row[metric1] and row[metric2]:
                try:
                    v1 = float(row[metric1])
                    v2 = float(row[metric2])
                    values1.append(v1)
                    values2.append(v2)
                except (ValueError, TypeError):
                    continue

        if len(values1) < 2:
            correlations.append({
                "metrics": pair,
                "error": "Insufficient data points"
            })
            continue

        # Calculate Pearson correlation
        n = len(values1)
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)

        numerator = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))
        denom1 = sum((v - mean1) ** 2 for v in values1) ** 0.5
        denom2 = sum((v - mean2) ** 2 for v in values2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            correlation = 0.0
        else:
            correlation = numerator / (denom1 * denom2)

        # Classify strength
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "negligible"

        correlations.append({
            "metrics": pair,
            "correlation": round(correlation, 3),
            "strength": strength,
            "n_samples": n
        })

    return {
        "run_id": run_id,
        "correlations": correlations
    }


async def get_metric_trend(
    run_id: str,
    metric: str,
    window: str = "all"
) -> Dict[str, Any]:
    """Analyze trend for a metric over a specified window."""
    import statistics

    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows or metric not in rows[0]:
        return {"error": f"Metric '{metric}' not found"}

    # Extract all values first
    all_values = []
    all_epochs = []
    for row in rows:
        if metric in row and row[metric]:
            try:
                val = float(row[metric])
                all_values.append(val)
                if "epoch" in row and row["epoch"]:
                    all_epochs.append(float(row["epoch"]))
            except (ValueError, TypeError):
                continue

    if not all_values:
        return {"error": f"No valid data for metric '{metric}'"}

    # Apply window filter
    if window == "all":
        values = all_values
        epochs = all_epochs
        window_desc = "entire training"
    elif window.startswith("last_") and window.endswith("_epochs"):
        try:
            n = int(window.replace("last_", "").replace("_epochs", ""))
            values = all_values[-n:]
            epochs = all_epochs[-n:] if all_epochs else []
            window_desc = f"last {n} epochs"
        except ValueError:
            return {"error": f"Invalid window format: {window}"}
    elif window.startswith("first_") and window.endswith("_epochs"):
        try:
            n = int(window.replace("first_", "").replace("_epochs", ""))
            values = all_values[:n]
            epochs = all_epochs[:n] if all_epochs else []
            window_desc = f"first {n} epochs"
        except ValueError:
            return {"error": f"Invalid window format: {window}"}
    else:
        return {"error": f"Invalid window format: {window}. Use 'all', 'last_N_epochs', or 'first_N_epochs'"}

    if len(values) < 2:
        return {"error": "Insufficient data in window"}

    # Calculate linear regression slope
    n = len(values)
    x = list(range(n))
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(values)
    numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    denominator = sum((xi - mean_x) ** 2 for xi in x)

    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator

    # Determine direction and significance
    if abs(slope) < 0.01:
        direction = "stable"
        significance = "low"
    elif slope > 0:
        direction = "improving"
        significance = "high" if slope > 0.1 else "moderate"
    else:
        direction = "declining"
        significance = "high" if slope < -0.1 else "moderate"

    # Calculate change
    change = values[-1] - values[0]
    change_pct = (change / abs(values[0]) * 100) if values[0] != 0 else 0.0

    return {
        "run_id": run_id,
        "metric": metric,
        "window": window_desc,
        "n_samples": n,
        "first_value": round(values[0], 4),
        "last_value": round(values[-1], 4),
        "change": round(change, 4),
        "change_pct": round(change_pct, 2),
        "slope": round(slope, 6),
        "direction": direction,
        "significance": significance,
        "epoch_range": [epochs[0], epochs[-1]] if epochs else None
    }


async def compare_to_baseline(
    run_id: str,
    baseline: str,
    metric: str = "train/roll/ep_rew/mean"
) -> Dict[str, Any]:
    """Compare a run's performance to a baseline."""

    # Load target run
    run = Run.load(run_id)
    run_config = run.load_config()
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows or metric not in rows[-1]:
        return {"error": f"Metric '{metric}' not found in run {run_id}"}

    try:
        target_value = float(rows[-1][metric])
        target_epoch = float(rows[-1].get("epoch", 0))
    except (ValueError, TypeError):
        return {"error": f"Invalid metric value in run {run_id}"}

    # Get baseline run
    if baseline == "best_for_env":
        best_result = await get_best_run(run_config.env_id, metric, minimize=False)
        if "error" in best_result:
            return {"error": f"Could not find best run: {best_result['error']}"}
        baseline_id = best_result["best_run"]["run_id"]
        baseline_value = best_result["best_value"]
        baseline_epoch = None
    else:
        # Specific run ID
        baseline_id = baseline
        try:
            baseline_run = Run.load(baseline_id)
            baseline_metrics = Path(baseline_run.run_dir) / "metrics.csv"

            if not baseline_metrics.exists():
                return {"error": f"No metrics for baseline run {baseline_id}"}

            with open(baseline_metrics) as f:
                reader = csv.DictReader(f)
                baseline_rows = list(reader)

            if not baseline_rows or metric not in baseline_rows[-1]:
                return {"error": f"Metric '{metric}' not found in baseline"}

            baseline_value = float(baseline_rows[-1][metric])
            baseline_epoch = float(baseline_rows[-1].get("epoch", 0))
        except Exception as e:
            return {"error": f"Failed to load baseline: {str(e)}"}

    # Calculate comparison
    difference = target_value - baseline_value
    if baseline_value != 0:
        pct_difference = (difference / abs(baseline_value)) * 100
    else:
        pct_difference = 0.0

    comparison = {
        "run_id": run_id,
        "baseline_id": baseline_id,
        "metric": metric,
        "target": {
            "value": round(target_value, 4),
            "epoch": target_epoch
        },
        "baseline": {
            "value": round(baseline_value, 4),
            "epoch": baseline_epoch
        },
        "difference": round(difference, 4),
        "pct_difference": round(pct_difference, 2),
        "status": "better" if difference > 0 else "worse" if difference < 0 else "equal"
    }

    return comparison


async def health_check(run_id: str) -> Dict[str, Any]:
    """Quick health check identifying top anomalies."""

    # Get run info
    run_info = await get_run_info(run_id)
    if "error" in run_info:
        return run_info

    config = Run.load(run_id).load_config()

    # Get alerts
    alerts_result = await get_metric_alerts(run_id)

    # Get key metric trends
    key_metrics = [
        "train/roll/ep_rew/mean",
        "train/opt/policy/entropy",
        "train/opt/value/explained_var"
    ]

    anomalies = []

    # Check alerts first
    if "alerts" in alerts_result and alerts_result["alerts"]:
        for alert in alerts_result["alerts"]:
            anomalies.append({
                "severity": "WARNING",
                "type": "metric_alert",
                "message": f"{alert['metric_alert']}: {alert['message']}",
                "tip": alert.get("tip", "")
            })

    # Check reward trend (handle missing metric gracefully)
    reward_trend = await get_metric_trend(run_id, "train/roll/ep_rew/mean", "last_20_epochs")
    if "error" not in reward_trend:
        if "direction" in reward_trend and reward_trend["direction"] == "declining":
            anomalies.append({
                "severity": "CRITICAL",
                "type": "reward_decline",
                "message": f"Rewards declining: {reward_trend['change_pct']:.1f}% over last 20 epochs",
                "tip": "Check if entropy is collapsing or learning rate is too high"
            })

    # Check entropy trend (should decrease)
    entropy_trend = await get_metric_trend(run_id, "train/opt/policy/entropy", "all")
    if "error" not in entropy_trend:
        if "direction" in entropy_trend and entropy_trend["direction"] == "improving":
            anomalies.append({
                "severity": "CRITICAL",
                "type": "entropy_collapse",
                "message": f"Entropy increasing instead of decreasing: {entropy_trend['change_pct']:.1f}% change",
                "tip": "Reduce ent_coef significantly (try 10x reduction)"
            })

    # Check explained variance
    expl_var_summary = await get_metrics_summary(run_id, "train/opt/value/explained_var")
    if "error" not in expl_var_summary and "last" in expl_var_summary:
        if expl_var_summary["last"] < 0.5:
            anomalies.append({
                "severity": "WARNING",
                "type": "poor_value_fit",
                "message": f"Explained variance low: {expl_var_summary['last']:.2f}",
                "tip": "Value function not learning well; check value_lr and architecture"
            })
        elif expl_var_summary["last"] > 0.98:
            anomalies.append({
                "severity": "INFO",
                "type": "potential_overfit",
                "message": f"Explained variance very high: {expl_var_summary['last']:.3f}",
                "tip": "May indicate value function overfitting"
            })

    # Check if solved
    reward_threshold = getattr(config, "reward_threshold", None)
    progress = await get_training_progress(run_id)
    is_solved = progress.get("performance", {}).get("is_solved", False)

    if not is_solved and reward_threshold:
        current = float(progress["performance"]["train_reward_mean"] or 0)
        gap = reward_threshold - current
        pct_gap = (gap / reward_threshold) * 100
        anomalies.append({
            "severity": "INFO",
            "type": "not_solved",
            "message": f"Not yet solved: {current:.1f}/{reward_threshold:.1f} ({pct_gap:.1f}% gap)",
            "tip": "Continue training or adjust hyperparameters"
        })

    # Sort by severity
    severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    anomalies.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return {
        "run_id": run_id,
        "anomaly_count": len(anomalies),
        "top_anomalies": anomalies[:5],  # Top 5
        "is_healthy": len([a for a in anomalies if a["severity"] == "CRITICAL"]) == 0
    }


async def plot_compare_runs(
    run_ids: List[str],
    metric: str,
    x_axis: str = "epoch",
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600
) -> list[types.ImageContent]:
    """Plot the same metric across multiple runs for comparison."""
    import base64
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Collect data from all runs
    run_data = {}

    for run_id in run_ids:
        try:
            run = Run.load(run_id)
            metrics_file = Path(run.run_dir) / "metrics.csv"

            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows or metric not in rows[0] or x_axis not in rows[0]:
                continue

            # Extract data
            x_data = []
            y_data = []

            for row in rows:
                if x_axis in row and metric in row and row[x_axis] and row[metric]:
                    try:
                        x_val = float(row[x_axis])
                        y_val = float(row[metric])
                        x_data.append(x_val)
                        y_data.append(y_val)
                    except (ValueError, TypeError):
                        continue

            if x_data and y_data:
                run_data[run_id] = (x_data, y_data)

        except Exception:
            continue

    if not run_data:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": "No valid data found for any run"})
        )]

    # Create plot
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

    for run_id, (x_data, y_data) in run_data.items():
        ax.plot(x_data, y_data, label=run_id, marker='o', markersize=2, alpha=0.8)

    ax.set_xlabel(x_axis)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save to buffer
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # Encode as base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return [types.ImageContent(
        type="image",
        data=img_base64,
        mimeType="image/png"
    )]


async def get_hyperparam_history(
    run_id: str,
    params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Track hyperparameter values across training.

    Returns summary only (first/last/min/max) to save tokens.
    Full history arrays are not included by default.
    """

    run = Run.load(run_id)
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"error": "No metrics data available"}

    # Auto-detect hyperparameter columns if not specified
    if params is None:
        # Look for common hyperparameter patterns in column names
        all_cols = list(rows[0].keys())
        hyperparam_patterns = ["_lr", "ent_coef", "clip_range", "gamma", "lam"]
        params = [col for col in all_cols if any(pattern in col for pattern in hyperparam_patterns)]

    if not params:
        return {"error": "No hyperparameters found or specified"}

    # Extract history (but only keep for analysis, don't return full arrays)
    history = {param: [] for param in params}
    first_epoch = None
    last_epoch = None

    for row in rows:
        if "epoch" in row and row["epoch"]:
            try:
                epoch = float(row["epoch"])
                if first_epoch is None:
                    first_epoch = epoch
                last_epoch = epoch
            except (ValueError, TypeError):
                pass

        for param in params:
            if param in row and row[param]:
                try:
                    history[param].append(float(row[param]))
                except (ValueError, TypeError):
                    pass

    # Check which params are scheduled (changing)
    scheduled = {}
    for param, values in history.items():
        if len(values) > 1:
            is_changing = len(set(values)) > 1
            scheduled[param] = {
                "is_scheduled": is_changing,
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "n_samples": len(values)
            }
        elif len(values) == 1:
            scheduled[param] = {
                "is_scheduled": False,
                "value": values[0],
                "n_samples": 1
            }
        else:
            scheduled[param] = {
                "is_scheduled": False,
                "value": None,
                "n_samples": 0
            }

    return {
        "run_id": run_id,
        "epoch_range": [first_epoch, last_epoch] if first_epoch is not None else None,
        "params": params,
        "scheduled_params": [p for p, info in scheduled.items() if info.get("is_scheduled", False)],
        "constant_params": [p for p, info in scheduled.items() if not info.get("is_scheduled", False)],
        "summary": scheduled
    }


async def comprehensive_diagnostic(
    run_id: str,
    include_recent_metrics: bool = True,
    include_full_config: bool = False
) -> Dict[str, Any]:
    """Comprehensive single-call diagnostic for run debugging.

    Consolidates status, config, progress, key metrics, health, and trends.
    Designed to minimize token usage while providing complete diagnostic picture.
    """
    # Resolve @last if needed
    if run_id == "@last":
        runs_result = await list_runs(limit=1)
        if "error" in runs_result or not runs_result.get("runs"):
            return {"error": "No runs found"}
        run_id = runs_result["runs"][0]["run_id"]

    run = Run.load(run_id)
    config = run.load_config()
    metrics_file = Path(run.run_dir) / "metrics.csv"

    if not metrics_file.exists():
        return {"error": f"No metrics file found for run {run_id}"}

    # Read metrics once
    with open(metrics_file) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        return {"error": "No metrics data available"}

    # Get training status
    status = await get_training_status(run_id)
    is_active = status.get("running", False)

    # Get first and last rows for analysis
    first_row = all_rows[0]
    last_row = all_rows[-1]

    # Recent metrics for trend analysis (last 20 epochs or less)
    recent_rows = all_rows[-20:] if include_recent_metrics else []

    # === PROGRESS ===
    total_env_steps = float(last_row.get("train/cnt/total_env_steps", 0))
    max_env_steps = getattr(config, "max_env_steps", None)
    progress_pct = (total_env_steps / max_env_steps * 100) if max_env_steps else None
    current_epoch = int(last_row.get("epoch", 0))

    # === PERFORMANCE ===
    train_reward = float(last_row.get("train/roll/ep_rew/mean", 0))
    val_reward = last_row.get("val/roll/ep_rew/mean", "")
    val_reward = float(val_reward) if val_reward else None
    best_reward = float(last_row.get("train/roll/ep_rew/best", 0))
    ep_len_mean = float(last_row.get("train/roll/ep_len/mean", 0))

    reward_threshold = getattr(config, "reward_threshold", None)
    is_solved = train_reward >= reward_threshold if reward_threshold else False
    gap_to_threshold = (reward_threshold - train_reward) if reward_threshold else None
    gap_pct = (gap_to_threshold / reward_threshold * 100) if reward_threshold else None

    # === KEY METRICS (current values) ===
    key_metrics = {
        "entropy": float(last_row.get("train/opt/policy/entropy", 0)),
        "approx_kl": float(last_row.get("train/opt/policy/approx_kl", 0)),
        "clip_frac": float(last_row.get("train/opt/policy/clip_frac", 0)),
        "explained_var": float(last_row.get("train/opt/value/explained_var", 0)),
        "policy_loss": float(last_row.get("train/opt/loss/policy", 0)),
        "value_loss": float(last_row.get("train/opt/loss/value", 0)),
        "value_clip_frac": float(last_row.get("train/opt/value/clip_frac", 0)),
        "grad_norm": float(last_row.get("train/opt/grad/norm", 0)),
    }

    # === TREND ANALYSIS (if recent metrics available) ===
    trends = {}
    if recent_rows and len(recent_rows) > 1:
        # Reward trend
        reward_values = [float(r.get("train/roll/ep_rew/mean", 0)) for r in recent_rows if r.get("train/roll/ep_rew/mean")]
        if len(reward_values) >= 2:
            reward_change = reward_values[-1] - reward_values[0]
            reward_change_pct = (reward_change / abs(reward_values[0]) * 100) if reward_values[0] != 0 else 0
            trends["reward"] = {
                "direction": "improving" if reward_change > 0 else "declining" if reward_change < 0 else "flat",
                "change": reward_change,
                "change_pct": reward_change_pct,
                "slope": reward_change / len(reward_values)
            }

        # Entropy trend
        entropy_values = [float(r.get("train/opt/policy/entropy", 0)) for r in recent_rows if r.get("train/opt/policy/entropy")]
        if len(entropy_values) >= 2:
            entropy_change = entropy_values[-1] - entropy_values[0]
            entropy_change_pct = (entropy_change / abs(entropy_values[0]) * 100) if entropy_values[0] != 0 else 0
            trends["entropy"] = {
                "direction": "increasing" if entropy_change > 0 else "decreasing" if entropy_change < 0 else "flat",
                "change": entropy_change,
                "change_pct": entropy_change_pct
            }

    # === HEALTH CHECK ===
    anomalies = []

    # Check KL divergence
    if key_metrics["approx_kl"] > 0.02:
        anomalies.append({
            "severity": "WARNING",
            "metric": "approx_kl",
            "value": key_metrics["approx_kl"],
            "message": f"KL divergence high: {key_metrics['approx_kl']:.4f} (threshold: 0.02)"
        })

    # Check explained variance
    if key_metrics["explained_var"] < 0.5:
        anomalies.append({
            "severity": "WARNING",
            "metric": "explained_var",
            "value": key_metrics["explained_var"],
            "message": f"Explained variance low: {key_metrics['explained_var']:.2f}"
        })

    # Check value clip fraction
    if key_metrics["value_clip_frac"] < 0.05:
        anomalies.append({
            "severity": "INFO",
            "metric": "value_clip_frac",
            "value": key_metrics["value_clip_frac"],
            "message": f"Value clip fraction low: {key_metrics['value_clip_frac']:.3f}"
        })

    # Check for reward decline
    if trends.get("reward", {}).get("direction") == "declining":
        anomalies.append({
            "severity": "CRITICAL",
            "metric": "reward_trend",
            "value": trends["reward"]["change"],
            "message": f"Rewards declining: {trends['reward']['change_pct']:.1f}% over recent epochs"
        })

    # === CONFIG SUMMARY ===
    if include_full_config:
        config_summary = config.__dict__
    else:
        config_summary = {
            "env_id": config.env_id,
            "algo": getattr(config, "algo", "unknown"),
            "n_envs": config.n_envs,
            "n_steps": getattr(config, "n_steps", None),
            "policy_lr": config.policy_lr,
            "batch_size": config.batch_size,
            "ent_coef": getattr(config, "ent_coef", None),
            "clip_range": getattr(config, "clip_range", None),
            "max_env_steps": max_env_steps,
        }

    # === TRAINING SPEED ===
    rollout_fps = float(last_row.get("train/time/rollout_fps", 0))
    system_fps = float(last_row.get("train/time/system_fps", 0))

    # === TIME ESTIMATES ===
    elapsed_time = None
    remaining_time_estimate = None
    if system_fps > 0 and max_env_steps:
        remaining_steps = max_env_steps - total_env_steps
        remaining_time_estimate = remaining_steps / system_fps  # seconds

        # Try to get elapsed time from run info
        run_info = await get_run_info(run_id)
        if "timestamp" in run_info:
            from datetime import datetime
            start_time = datetime.fromisoformat(run_info["timestamp"])
            elapsed_time = (datetime.now() - start_time).total_seconds()

    return {
        "run_id": run_id,
        "status": {
            "is_active": is_active,
            "is_solved": is_solved,
            "health": "healthy" if len([a for a in anomalies if a["severity"] == "CRITICAL"]) == 0 else "issues_detected"
        },
        "progress": {
            "total_env_steps": int(total_env_steps),
            "max_env_steps": max_env_steps,
            "progress_pct": round(progress_pct, 2) if progress_pct else None,
            "current_epoch": current_epoch,
            "elapsed_seconds": round(elapsed_time) if elapsed_time else None,
            "remaining_seconds_estimate": round(remaining_time_estimate) if remaining_time_estimate else None
        },
        "performance": {
            "train_reward_mean": round(train_reward, 2),
            "val_reward_mean": round(val_reward, 2) if val_reward else None,
            "best_reward": round(best_reward, 2),
            "ep_len_mean": round(ep_len_mean, 1),
            "reward_threshold": reward_threshold,
            "gap_to_threshold": round(gap_to_threshold, 2) if gap_to_threshold else None,
            "gap_pct": round(gap_pct, 2) if gap_pct else None
        },
        "key_metrics": {k: round(v, 4) for k, v in key_metrics.items()},
        "trends": {k: {**v, "change": round(v["change"], 4), "change_pct": round(v["change_pct"], 2)} for k, v in trends.items()} if trends else None,
        "anomalies": anomalies,
        "config": config_summary,
        "training_speed": {
            "rollout_fps": round(rollout_fps, 1),
            "system_fps": round(system_fps, 1)
        }
    }


async def run_play(
    run_id: Optional[str] = None,
    config_id: Optional[str] = None,
    episodes: int = 1,
    deterministic: bool = False,
    headless: bool = True,
    mode: Optional[str] = None,
    seed: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    show_preprocessing: bool = False,
    show_cnn_filters: bool = False
) -> Dict[str, Any]:
    """Play a trained policy or test environment with random/user policy."""

    # Build command
    cmd = ["python", "run_play.py"]

    # Add run_id or config_id
    if run_id and config_id:
        return {"error": "Cannot specify both run_id and config_id"}
    elif run_id:
        cmd.extend(["--run-id", run_id])
    elif config_id:
        cmd.extend(["--config-id", config_id])
    else:
        return {"error": "Must specify either run_id or config_id"}

    # Add options
    if episodes != 1:
        cmd.extend(["--episodes", str(episodes)])

    if deterministic:
        cmd.append("--deterministic")

    if headless:
        cmd.append("--headless")

    if mode:
        cmd.extend(["--mode", mode])

    if seed:
        cmd.extend(["--seed", str(seed)])

    if show_preprocessing:
        cmd.append("--show-preprocessing")

    if show_cnn_filters:
        cmd.append("--show-cnn-filters")

    # Add env_kwargs
    if env_kwargs:
        for key, value in env_kwargs.items():
            cmd.extend(["--env-kwargs", f"{key}={value}"])

    # Run synchronously and capture output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out after 5 minutes",
            "command": " ".join(cmd)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "command": " ".join(cmd)
        }


async def run_inspect(
    run_id: str = "@last",
    port: int = 7860,
    host: str = "localhost",
    share: bool = False,
    seed: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Launch Gradio inspection UI in background."""

    # Build command
    cmd = ["python", "run_inspect.py"]

    if run_id != "@last":
        cmd.extend(["--run-id", run_id])

    if port != 7860:
        cmd.extend(["--port", str(port)])

    if host != "localhost":
        cmd.extend(["--host", host])

    if share:
        cmd.append("--share")

    if seed:
        cmd.extend(["--seed", str(seed)])

    # Add env_kwargs
    if env_kwargs:
        for key, value in env_kwargs.items():
            cmd.extend(["--env-kwargs", f"{key}={value}"])

    # Start process in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a moment to check if it started successfully
    await asyncio.sleep(2)

    if process.poll() is not None:
        # Process already exited
        stdout, stderr = process.communicate()
        return {
            "success": False,
            "error": "Gradio UI exited immediately",
            "stdout": stdout,
            "stderr": stderr,
            "command": " ".join(cmd)
        }

    # Track process with a unique key (use port as identifier)
    inspect_key = f"inspect_{port}"
    _running_processes[inspect_key] = process

    url = f"http://{host}:{port}"

    return {
        "success": True,
        "run_id": run_id,
        "pid": process.pid,
        "url": url,
        "message": f"Gradio UI started at {url}. Use stop_training with run_id='{inspect_key}' to stop.",
        "command": " ".join(cmd)
    }


async def run_publish(
    run_id: str = "@last",
    repo: Optional[str] = None,
    private: bool = False
) -> Dict[str, Any]:
    """Publish a training run to Hugging Face Hub."""

    # Build command
    cmd = ["python", "run_publish.py"]

    if run_id != "@last":
        cmd.extend(["--run-id", run_id])

    if repo:
        cmd.extend(["--repo", repo])

    if private:
        cmd.append("--private")

    # Run synchronously and capture output
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for uploads
        )

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "command": " ".join(cmd)
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out after 10 minutes",
            "command": " ".join(cmd)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "command": " ".join(cmd)
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
