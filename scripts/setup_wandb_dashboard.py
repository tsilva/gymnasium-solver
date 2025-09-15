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
    p.add_argument("--name", default=None, help="Workspace name (title in UI). Defaults to '<project> View'.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite workspace with the same name if it exists")
    p.add_argument(
        "--key-panels-per-section",
        type=int,
        default=12,
        help="When many key metrics exist, split them across multiple sections with at most this many panels each.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the spec and exit without pushing")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.entity or not args.project:
        print("Error: --entity and --project are required (or set WANDB_ENTITY/WANDB_PROJECT)", file=sys.stderr)
        return 2
    # Default workspace name to '<project> View' when not provided
    if not args.name:
        args.name = f"{args.project} View"

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
        # Create one panel per metric for clarity, then split into multiple sections to avoid manual resizing
        def _chunks(seq: list[str], size: int) -> list[list[str]]:
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        for idx, group in enumerate(_chunks(key_metrics, max(args.key_panels_per_section, 1)), start=1):
            panels = []
            for m in group:
                panels.append(
                    wr.LinePlot(
                        title=m,
                        x=default_x,
                        y=[m],
                    )
                )

            # Name first section plainly; later ones include index range for clarity
            if idx == 1:
                name = "Key Metrics"
            else:
                start = (idx - 1) * args.key_panels_per_section + 1
                end = start + len(group) - 1
                name = f"Key Metrics {start}–{end}"

            sections.append(
                ws.Section(
                    name=name,
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

    # If a workspace with this display name already exists, respect --overwrite
    try:
        from wandb_gql import gql  # lightweight GQL helper shipped with wandb

        query = gql(
            """
            query ProjectViews($entityName: String, $projectName: String, $viewType: String = "project-view") {
              project(name: $projectName, entityName: $entityName) {
                allViews(viewType: $viewType) {
                  edges { node { id name displayName } }
                }
              }
            }
            """
        )

        resp = api.client.execute(
            query,
            {"entityName": args.entity, "projectName": args.project, "viewType": "project-view"},
        )
        edges = (((resp or {}).get("project") or {}).get("allViews") or {}).get("edges", [])
        existing = None
        for e in edges:
            node = e.get("node") or {}
            if node.get("displayName") == args.name:
                existing = node
                break

        if existing and not args.overwrite:
            # Construct URL to the existing workspace
            try:
                from wandb_workspaces.workspaces import internal as ws_internal
                base = wandb.Api().client.app_url.rstrip("/")
                nw = ws_internal._internal_name_to_url_query_str(existing.get("name", ""))
                url = f"{base}/{args.entity}/{args.project}?nw={nw}"
            except Exception:
                url = None
            msg = f"Workspace '{args.name}' already exists under {args.entity}/{args.project}."
            if url:
                msg += f" URL: {url}"
            print(msg)
            print("Pass --overwrite to update this workspace.")
            return 0
    except Exception as e:
        # If we can't check, be conservative and only proceed when --overwrite is set
        if not args.overwrite:
            print(
                f"Could not determine if workspace '{args.name}' exists (reason: {e}). Add --overwrite to update/create explicitly.",
                file=sys.stderr,
            )
            return 2

    # Create new or update existing (when --overwrite)
    try:
        # If overwriting an existing view, fetch it by URL so we preserve its internal id
        saved_ws = None
        if args.overwrite:
            try:
                from wandb_workspaces.workspaces import internal as ws_internal
                # Attempt to find the internal name for the display name again
                # Reuse edges from previous call if available; otherwise, refetch
                internal_name = None
                try:
                    internal_name = existing.get("name") if existing else None
                except Exception:
                    internal_name = None
                if internal_name is None:
                    # Refetch minimal if needed
                    internal_name = None
                    try:
                        resp2 = api.client.execute(
                            query, {"entityName": args.entity, "projectName": args.project, "viewType": "project-view"}
                        )
                        edges2 = (
                            (((resp2 or {}).get("project") or {}).get("allViews") or {}).get("edges", [])
                        )
                        for e in edges2:
                            node = e.get("node") or {}
                            if node.get("displayName") == args.name:
                                internal_name = node.get("name")
                                break
                    except Exception:
                        internal_name = None
                if internal_name:
                    base = wandb.Api().client.app_url.rstrip("/")
                    nw = ws_internal._internal_name_to_url_query_str(internal_name)
                    url = f"{base}/{args.entity}/{args.project}?nw={nw}"
                    existing_ws = ws.Workspace.from_url(url)
                    # Apply the new sections/settings
                    existing_ws.sections = sections
                    saved_ws = existing_ws.save()
            except Exception:
                # Fallback to creating/saving without explicit id
                pass
        if saved_ws is None:
            saved_ws = workspace.save()
    except Exception as e:
        try:
            import requests  # type: ignore
            if isinstance(e, requests.exceptions.HTTPError):
                print(
                    "Error pushing workspace (HTTP error). Check WANDB login and WANDB_BASE_URL (should not include /graphql).",
                    file=sys.stderr,
                )
        except Exception:
            pass
        print(f"Save failed: {e}", file=sys.stderr)
        return 2

    # Print a friendly link
    try:
        print(f"Workspace saved: {saved_ws.url}")
    except Exception:
        print("Workspace saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
