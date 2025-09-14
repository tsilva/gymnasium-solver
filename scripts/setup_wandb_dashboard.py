"""
Create and push a W&B Workspace (dashboard) with common RL panels.

Usage:
  python scripts/setup_wandb_dashboard.py \
    --entity <your-wandb-entity> \
    --project <your-project> \
    --name "Gymnasium Solver Workspace"

Notes:
- Requires: `pip install wandb wandb-workspaces`
- Reads defaults from env: WANDB_ENTITY, WANDB_PROJECT, WANDB_API_KEY(optional)
- Use --dry-run to print the workspace spec without pushing to W&B.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a W&B Workspace and push it.")
    p.add_argument("--entity", default=os.getenv("WANDB_ENTITY"), help="W&B entity (team or username)")
    p.add_argument("--project", default=os.getenv("WANDB_PROJECT"), help="W&B project name")
    p.add_argument("--name", default="Gymnasium Solver Workspace", help="Workspace name (title in UI)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite workspace with the same name if it exists")
    p.add_argument("--dry-run", action="store_true", help="Print the spec and exit without pushing")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.entity or not args.project:
        print("Error: --entity and --project are required (or set WANDB_ENTITY/WANDB_PROJECT)", file=sys.stderr)
        return 2

    try:
        import wandb  # noqa: F401
    except Exception as e:
        print("Error: wandb is required. pip install wandb", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 2

    try:
        # Workspaces (containers) and Panels (visualizations)
        from wandb_workspaces import workspaces as ws
        # Newer versions expose panel types under the Reports v2 interface
        import wandb_workspaces.reports.v2.interface as wr
    except Exception as e:
        print("Error: wandb-workspaces is required. pip install wandb-workspaces", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        return 2

    # Sanitize WANDB_BASE_URL if it mistakenly includes /graphql
    base_url = os.environ.get("WANDB_BASE_URL")
    if base_url and base_url.rstrip("/").endswith("/graphql"):
        os.environ["WANDB_BASE_URL"] = base_url.rstrip("/")[:-len("/graphql")]

    # We'll ensure the target project exists just before pushing (skip for --dry-run)

    # Try to source key metrics order from config/metrics.yaml (_global.key_priority)
    key_metrics = None  # type: ignore[assignment]
    try:
        # Resolve repo root from this script location
        repo_root = Path(__file__).resolve().parents[1]
        metrics_yaml = repo_root / "config" / "metrics.yaml"

        # Prefer project IO helper; fallback to bare YAML if unavailable
        try:
            from utils.io import read_yaml  # type: ignore
            metrics_cfg = read_yaml(metrics_yaml)
        except Exception:  # pragma: no cover - best-effort fallback
            import yaml  # type: ignore

            with metrics_yaml.open("r", encoding="utf-8") as f:
                metrics_cfg = yaml.safe_load(f)

        global_cfg = metrics_cfg.get("_global", {}) if isinstance(metrics_cfg, dict) else {}
        # Accept both key names for robustness
        key_metrics = global_cfg.get("key_priority") or global_cfg.get("keypriority")
        if key_metrics is not None:
            # Normalize to strings and keep order
            key_metrics = [str(k) for k in key_metrics if k is not None]
    except Exception as e:
        print(
            f"Warning: failed to read key metrics from config/metrics.yaml: {e}. Using defaults.",
            file=sys.stderr,
        )
        key_metrics = None

    # Use a conservative default x-axis that always exists in W&B
    default_x = "Step"

    # Compose sections. Always include a Key Metrics section when key_priority is found.
    sections = []
    if key_metrics:
        # Large lists can create heavy payloads; split into chunks to keep panels responsive
        def _chunks(seq: list[str], size: int) -> list[list[str]]:
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        chunks = _chunks(key_metrics, 15)
        panels = []
        for idx, ys in enumerate(chunks, start=1):
            start = (idx - 1) * 15 + 1
            end = start + len(ys) - 1
            panels.append(
                wr.LinePlot(
                    title=f"Key Metrics {start}–{end}",
                    x=default_x,
                    y=ys,
                )
            )

        sections.append(
            ws.Section(
                name="Key Metrics",
                is_open=True,
                panels=panels,
            )
        )

    # Keep a couple of focused panels for quick-glance diagnostics
    sections.extend(
        [
            ws.Section(
                name="Diagnostics",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Policy LR (scheduled)",
                        x=default_x,
                        y=["train/policy_lr"],
                    ),
                    wr.LinePlot(
                        title="Losses",
                        x=default_x,
                        y=["train/policy_loss", "train/value_loss", "train/entropy"],
                    ),
                    wr.ScalarChart(
                        title="Best Eval Reward",
                        metric="val/ep_rew_best",
                        groupby_aggfunc="max",
                    ),
                ],
            ),
            ws.Section(
                name="Videos",
                is_open=False,
                panels=[
                    # W&B video/media panels generally auto-populate when media with keys like
                    # "train/episodes" or "val/episodes" are logged. We add a placeholder here
                    # so users can pin recent episodes.
                    # In Reports v2, use MediaBrowser with media_keys
                    wr.MediaBrowser(
                        title="Recent Episodes", media_keys=["train/episodes", "val/episodes"]
                    ),
                ],
            ),
        ]
    )

    workspace = ws.Workspace(
        name=args.name,
        entity=args.entity,
        project=args.project,
        sections=sections,
    )

    if args.dry_run:
        # Print the JSON payload that would be sent
        try:
            as_json = workspace.to_json()  # type: ignore[attr-defined]
        except Exception:
            # Fallback if the version doesn't expose to_json()
            as_json = {
                "name": args.name,
                "entity": args.entity,
                "project": args.project,
                "sections": [
                    {
                        "name": s.name,
                        "is_open": getattr(s, "is_open", True),
                        "panels": [getattr(p, "title", type(p).__name__) for p in getattr(s, "panels", [])],
                    }
                    for s in sections
                ],
            }
        import json

        print(json.dumps(as_json, indent=2))
        return 0

    # Ensure the target project exists (Workspaces API doesn't auto-create projects)
    try:
        api = wandb.Api()
        projects = list(api.projects(args.entity))
        if not any(getattr(p, "name", None) == args.project for p in projects):
            api.create_project(args.project, args.entity)
    except Exception as e:
        print(
            f"Error ensuring project {args.entity}/{args.project} exists: {e}",
            file=sys.stderr,
        )
        return 2

    # Push to W&B
    # Some versions of wandb-workspaces do not accept the `overwrite` kwarg.
    # Try with the kwarg first for forward-compatibility; on TypeError, retry without it.
    try:
        saved = workspace.save(overwrite=args.overwrite)
    except TypeError:
        if args.overwrite:
            print(
                "Warning: this version of wandb-workspaces does not support --overwrite; proceeding without it.",
                file=sys.stderr,
            )
        saved = workspace.save()
    except Exception as e:
        import requests

        # If we hit an HTTP 404 to /graphql, suggest base URL/auth issues and exit with context
        if isinstance(e, requests.exceptions.HTTPError):
            print(
                "Error pushing workspace (HTTP error). Check WANDB login and WANDB_BASE_URL (should not include /graphql).",
                file=sys.stderr,
            )
        print(f"Save failed: {e}", file=sys.stderr)
        return 2

    # Try to print a friendly link if available
    url = getattr(saved, "url", None) or getattr(workspace, "url", None)
    print("Workspace saved.")
    if url:
        print(f"URL: {url}")
    else:
        print("Note: Workspace URL not provided by the library version.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
