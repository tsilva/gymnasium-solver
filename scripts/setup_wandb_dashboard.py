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

    # Define a small, sensible default dashboard for RL
    # - Reward curves vs total timesteps
    # - Best eval reward scalar
    # - Training diagnostic metrics
    sections = [
        ws.Section(
            name="Rewards",
            is_open=True,
            panels=[
                # Line plot showing train/eval reward over time
                wr.LinePlot(
                    title="Episode Reward",
                    x="train/total_timesteps",
                    y=["train/ep_rew_mean", "val/ep_rew_mean"],
                ),
                # Scalar showing the running best eval reward
                wr.ScalarChart(title="Best Eval Reward", metric="val/ep_rew_best", groupby_aggfunc="max"),
            ],
        ),
        ws.Section(
            name="Diagnostics",
            is_open=True,
            panels=[
                wr.LinePlot(
                    title="Policy LR (scheduled)",
                    x="train/total_timesteps",
                    y=["train/policy_lr"],
                ),
                wr.LinePlot(
                    title="Losses",
                    x="train/total_timesteps",
                    y=["train/policy_loss", "train/value_loss", "train/entropy"],
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
                wr.MediaBrowser(title="Recent Episodes", media_keys=["train/episodes", "val/episodes"]),
            ],
        ),
    ]

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

    # Push to W&B
    saved = workspace.save(overwrite=args.overwrite)

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
