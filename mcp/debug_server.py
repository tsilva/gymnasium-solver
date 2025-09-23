"""Lightweight MCP debug server for gymnasium-solver.

This server exposes a handful of tools that speed up iterative debugging:
- List recent runs with key metrics (`list_runs`).
- Summarise a specific run (`summarize_run`).
- Inspect the tail of a metric timeseries without loading the full CSV (`metrics_tail`).
- Slice run configuration with optional key filtering (`config_slice`).
- Probe object-based Atari environments to verify OCAtari outputs (`probe_objects`).

The server speaks a minimal JSON-RPC protocol over stdio:
- Request:  {"id": <int|str>, "method": "list_tools"}
- Request:  {"id": <int|str>, "method": "call_tool", "params": {"name": "tool", "args": {...}}}
- Response: {"id": <same>, "result": ...} or {"id": <same>, "error": {"message": str}}

See ``guides/mcp_debug_server.md`` for usage details.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from utils.environment import build_env
@dataclass
class Tool:
    """Metadata for a callable MCP tool."""

    name: str
    description: str
    params_schema: Dict[str, Any]
    handler: Callable[..., Any]

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params_schema,
        }


class MCPDebugServer:
    """Expose repo-aware debugging helpers through a minimal MCP interface."""

    def __init__(self, *, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or Path.cwd()
        self.runs_root = self.repo_root / "runs"
        self._tools: Dict[str, Tool] = {}
        self._register_tools()

    # -----------------
    # Public entrypoints
    # -----------------
    def run_stdio(self) -> None:
        """Process newline-delimited JSON requests from stdin and reply on stdout."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle_request(request)
            except Exception as exc:  # noqa: BLE001 - top-level guard to keep server alive
                response = {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "error": {"message": str(exc)},
                }
            sys.stdout.write(json.dumps(response) + os.linesep)
            sys.stdout.flush()

    # -----------------
    # Tool registration
    # -----------------
    def _register_tools(self) -> None:
        self._register(
            Tool(
                name="list_runs",
                description="List recent runs with coarse metrics (sorted by mtime desc).",
                params_schema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": ["string", "null"], "description": "Filter by config project_id."},
                        "limit": {"type": "integer", "minimum": 1, "default": 5},
                        "include_stats": {"type": "boolean", "default": True},
                    },
                    "additionalProperties": False,
                },
                handler=self.list_runs,
            )
        )

        self._register(
            Tool(
                name="summarize_run",
                description="Summarize a run: config highlights, reward stats, KL, clip fraction.",
                params_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                    },
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
                handler=self.summarize_run,
            )
        )

        self._register(
            Tool(
                name="metrics_tail",
                description="Return the last N samples of a metric timeseries from metrics.csv.",
                params_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "metric": {"type": "string", "description": "Metric key e.g. 'train/ep_rew_mean'."},
                        "limit": {"type": "integer", "minimum": 1, "default": 5},
                        "stage": {
                            "type": "string",
                            "default": "train",
                            "description": "Stage prefix if metric omits it (train/val/test).",
                        },
                    },
                    "required": ["run_id", "metric"],
                    "additionalProperties": False,
                },
                handler=self.metrics_tail,
            )
        )

        self._register(
            Tool(
                name="config_slice",
                description="Return config.json (optionally filtered by keys) for quick inspection.",
                params_schema={
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional whitelist of config keys to return.",
                        },
                    },
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
                handler=self.config_slice,
            )
        )

        self._register(
            Tool(
                name="probe_objects",
                description="Step an OCAtari env briefly to verify objects + feature vector alignment.",
                params_schema={
                    "type": "object",
                    "properties": {
                        "env_id": {"type": "string", "default": "ALE/Pong-v5"},
                        "obs_type": {"type": "string", "default": "objects"},
                        "n_steps": {"type": "integer", "minimum": 1, "default": 10},
                        "seed": {"type": ["integer", "null"], "default": 42},
                        "actions": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional sequence of discrete actions (fallback: serve + zeros).",
                        },
                        "include_features": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
                handler=self.probe_objects,
            )
        )

    def _register(self, tool: Tool) -> None:
        assert tool.name not in self._tools, f"Tool already registered: {tool.name}"
        self._tools[tool.name] = tool

    # ------------
    # Request core
    # ------------
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("Request must be a JSON object")
        req_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str):
            raise ValueError("Missing method")

        if method == "list_tools":
            result = {"tools": [tool.metadata() for tool in self._tools.values()]}
        elif method == "call_tool":
            params = request.get("params") or {}
            if not isinstance(params, dict):
                raise TypeError("params must be an object")
            name = params.get("name")
            if not isinstance(name, str):
                raise ValueError("Tool name must be provided")
            args = params.get("args") or {}
            if not isinstance(args, dict):
                raise TypeError("args must be an object")
            result = self._call_tool(name, args)
        else:
            raise ValueError(f"Unknown method: {method}")

        return {"id": req_id, "result": result}

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool '{name}'")
        handler = self._tools[name].handler
        return handler(**args)

    # ---------
    # Tool impl
    # ---------
    def list_runs(
        self,
        *,
        project_id: Optional[str] = None,
        limit: int = 5,
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        runs = self._iter_runs(project_id=project_id)
        limited = list(runs)[:limit]
        summaries = [self._summarize_run(run) for run in limited]
        payload = []
        for summary in summaries:
            row = {
                "run_id": summary["run_id"],
                "mtime": summary["mtime"],
                "env_id": summary.get("env_id"),
                "algo_id": summary.get("algo_id"),
                "total_timesteps": summary.get("total_timesteps"),
            }
            if include_stats:
                row.update(
                    {
                        "train": summary.get("train"),
                        "val": summary.get("val"),
                    }
                )
            payload.append(row)
        return {"runs": payload}

    def summarize_run(self, *, run_id: str) -> Dict[str, Any]:
        summary = self._summarize_run(run_id)
        return summary

    def metrics_tail(
        self,
        *,
        run_id: str,
        metric: str,
        limit: int = 5,
        stage: str = "train",
    ) -> Dict[str, Any]:
        metric_key = metric if "/" in metric else f"{stage}/{metric}"
        rows = self._read_metrics(run_id)
        samples = []
        for row in reversed(rows):
            value_raw = row.get(metric_key)
            value = _safe_float(value_raw)
            if value is None:
                continue
            total_ts = _safe_float(row.get("train/total_timesteps"))
            if total_ts is None:
                total_ts = _safe_float(row.get("total_timesteps"))
            sample = {
                "epoch": row.get("epoch"),
                "total_timesteps": total_ts,
                "value": value,
            }
            samples.append(sample)
            if len(samples) >= limit:
                break
        samples.reverse()
        stats = None
        if samples:
            values = [s["value"] for s in samples]
            stats = {
                "min": min(values),
                "max": max(values),
                "mean": fmean(values),
            }
        return {"metric": metric_key, "samples": samples, "stats": stats}

    def config_slice(self, *, run_id: str, keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        config_path = self._run_path(run_id) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json for run {run_id}")
        with config_path.open() as fh:
            config = json.load(fh)
        if keys:
            key_set = set(keys)
            filtered = {k: config.get(k) for k in key_set}
        else:
            filtered = config
        return {"run_id": run_id, "config": filtered}

    def probe_objects(
        self,
        *,
        env_id: str = "ALE/Pong-v5",
        obs_type: str = "objects",
        n_steps: int = 10,
        seed: Optional[int] = 42,
        actions: Optional[Iterable[int]] = None,
        include_features: bool = False,
    ) -> Dict[str, Any]:
        env = build_env(
            env_id,
            n_envs=1,
            obs_type=obs_type,
            env_wrappers=[{"id": "PongV5_FeatureExtractor"}]
            if obs_type == "objects"
            else [],
            subproc=False,
            render_mode=None,
        )
        try:
            if seed is not None and hasattr(env, "seed"):
                env.seed(seed)
            obs = env.reset()
            action_seq = list(actions) if actions is not None else self._default_action_sequence(env, n_steps)
            snapshots = []
            oc_env = _find_env_with_attr(env, "objects")
            if oc_env is None:
                raise RuntimeError("Could not locate an OCAtari-compatible env with 'objects' attribute")

            for step_idx in range(n_steps):
                action = action_seq[step_idx] if step_idx < len(action_seq) else action_seq[-1]
                action_arr = np.asarray([action], dtype=np.int64)
                next_obs, rewards, dones, infos = env.step(action_arr)
                objects_payload = [
                    _object_snapshot(obj)
                    for obj in getattr(oc_env, "objects", [])
                ]
                snapshot = {
                    "step": step_idx,
                    "action": int(action),
                    "reward": float(rewards[0]) if len(rewards) else 0.0,
                    "done": bool(dones[0]) if len(dones) else False,
                    "objects": objects_payload,
                }
                if include_features:
                    snapshot["features"] = _to_list(next_obs)
                snapshots.append(snapshot)
                obs = next_obs
            unique_categories = sorted({obj["category"] for snap in snapshots for obj in snap["objects"]})
            return {
                "env_id": env_id,
                "obs_type": obs_type,
                "n_steps": len(snapshots),
                "unique_categories": unique_categories,
                "snapshots": snapshots,
            }
        finally:
            env.close()

    # -----------------
    # Internal helpers
    # -----------------
    def _iter_runs(self, *, project_id: Optional[str]) -> Iterable[str]:
        if not self.runs_root.exists():
            return []
        entries = []
        for entry in self.runs_root.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("@"):
                continue
            if project_id is not None:
                config_path = entry / "config.json"
                if not config_path.exists():
                    continue
                try:
                    with config_path.open() as fh:
                        config = json.load(fh)
                except Exception:
                    continue
                if config.get("project_id") != project_id:
                    continue
            entries.append((entry.name, entry.stat().st_mtime))
        entries.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in entries]

    @lru_cache(maxsize=128)
    def _summarize_run(self, run_id: str) -> Dict[str, Any]:
        run_path = self._run_path(run_id)
        mtime = run_path.stat().st_mtime
        config = self.config_slice(run_id=run_id)["config"]
        metrics_rows = self._read_metrics(run_id)
        totals = [
            _safe_float(row.get("train/total_timesteps"))
            or _safe_float(row.get("total_timesteps"))
            for row in metrics_rows
        ]
        total_timesteps = max((value for value in totals if value is not None), default=None)

        def _latest_metric(prefix: str, key: str) -> Optional[float]:
            metric_key = f"{prefix}/{key}"
            for row in reversed(metrics_rows):
                value = _safe_float(row.get(metric_key))
                if value is not None:
                    return value
            return None

        def _best_metric(prefix: str, key: str) -> Optional[float]:
            metric_key = f"{prefix}/{key}"
            best = None
            for row in metrics_rows:
                value = _safe_float(row.get(metric_key))
                if value is None:
                    continue
                best = value if best is None else max(best, value)
            return best

        train_block = {
            "ep_rew_mean": _latest_metric("train", "ep_rew_mean"),
            "ep_rew_best": _best_metric("train", "ep_rew_mean"),
            "clip_fraction": _latest_metric("train", "clip_fraction"),
            "approx_kl": _latest_metric("train", "approx_kl"),
        }
        val_block = None
        if any("val/" in key for key in metrics_rows[-1].keys()):
            val_block = {
                "ep_rew_mean": _latest_metric("val", "ep_rew_mean"),
                "ep_rew_best": _best_metric("val", "ep_rew_mean"),
            }

        return {
            "run_id": run_id,
            "mtime": mtime,
            "env_id": config.get("env_id"),
            "algo_id": config.get("algo_id"),
            "total_timesteps": total_timesteps,
            "train": train_block,
            "val": val_block,
        }

    @lru_cache(maxsize=128)
    def _read_metrics(self, run_id: str) -> List[Dict[str, Any]]:
        metrics_path = self._run_path(run_id) / "metrics.csv"
        if not metrics_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with metrics_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
        return rows

    def _run_path(self, run_id: str) -> Path:
        path = self.runs_root / run_id
        if not path.exists():
            raise FileNotFoundError(f"Unknown run: {run_id}")
        return path

    def _default_action_sequence(self, env, n_steps: int) -> List[int]:
        # Try to serve once then stay still (assuming Pong layout); fallback to zeros.
        try:
            serve_action = 1  # Fire
            idle_action = 0
            return [serve_action] + [idle_action] * (n_steps - 1)
        except Exception:
            return [0] * n_steps


# ---------
# Utilities
# ---------

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_list(obs: Any) -> List[float]:
    if isinstance(obs, (list, tuple)):
        return [float(x) for x in obs[0]] if obs and isinstance(obs[0], (list, tuple, np.ndarray)) else [float(x) for x in obs]
    if isinstance(obs, np.ndarray):
        return obs.astype(float).flatten().tolist()
    return [float(obs)]


def _find_env_with_attr(root_env: Any, attr: str) -> Optional[Any]:
    visited = set()
    stack = [root_env]
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if hasattr(current, attr):
            return current
        for name in ("env", "venv"):
            child = getattr(current, name, None)
            if child is not None:
                stack.append(child)
        if hasattr(current, "envs"):
            stack.extend(list(getattr(current, "envs")))
    return None


def _object_snapshot(obj: Any) -> Dict[str, Any]:
    def _maybe_get(o: Any, name: str, default: Any = None) -> Any:
        return getattr(o, name, default)

    payload = {
        "category": str(_maybe_get(obj, "category", "?")),
        "center": tuple(map(float, _maybe_get(obj, "center", (0.0, 0.0)))) if hasattr(obj, "center") else None,
        "x": float(_maybe_get(obj, "x", 0.0)),
        "y": float(_maybe_get(obj, "y", 0.0)),
        "w": float(_maybe_get(obj, "w", 0.0)),
        "h": float(_maybe_get(obj, "h", 0.0)),
        "dx": float(_maybe_get(obj, "dx", 0.0)),
        "dy": float(_maybe_get(obj, "dy", 0.0)),
        "visible": bool(_maybe_get(obj, "visible", True)),
    }
    return payload


# -----
# CLI
# -----

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MCP debug server over stdio")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (defaults to current working directory)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    server = MCPDebugServer(repo_root=args.repo_root)
    server.run_stdio()


if __name__ == "__main__":
    main()
