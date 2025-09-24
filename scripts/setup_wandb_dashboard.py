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

Implementation detail: Logic lives in `utils.wandb_workspace` and is reused
by training to create/update a workspace at the beginning of a run.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on sys.path when running as a script from scripts/
import sys as _sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from utils.wandb_workspace import WorkspaceRequest, create_or_update_workspace


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
    p.add_argument("--select-run-id", default=None, help="Default-select only this run id across panels")
    p.add_argument("--select-latest", action="store_true", help="Fetch most recent run in the project and select it by default")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.entity or not args.project:
        print("Error: --entity and --project are required (or set WANDB_ENTITY/WANDB_PROJECT)", file=sys.stderr)
        return 2
    # Default workspace name to '<project> View' when not provided
    if not args.name:
        args.name = f"{args.project} View"

    # Optional: resolve --select-latest into a concrete run id
    select_run_id = args.select_run_id
    if args.select_latest and not select_run_id:
        try:
            import wandb
            api = wandb.Api()
            runs = list(api.runs(f"{args.entity}/{args.project}"))
            if runs:
                # Choose most recent by created_at
                runs.sort(key=lambda r: getattr(r, "created_at", None) or 0, reverse=True)
                select_run_id = runs[0].id
        except Exception as e:
            print(f"Warning: could not resolve --select-latest ({e}). Falling back to all runs.")

    # Dry-run returns JSON spec to stdout; otherwise push and print URL
    if args.dry_run:
        try:
            json_spec = create_or_update_workspace(
                WorkspaceRequest(
                    entity=args.entity,
                    project=args.project,
                    name=args.name or f"{args.project} View",
                    key_panels_per_section=max(args.key_panels_per_section, 1),
                    overwrite=False,
                    dry_run=True,
                    select_run_id=select_run_id,
                )
            )
            print(json_spec)
            return 0
        except ImportError as e:
            print(str(e), file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Error generating workspace spec: {e}", file=sys.stderr)
            return 2

    # Push workspace and print resulting URL
    try:
        url = create_or_update_workspace(
            WorkspaceRequest(
                entity=args.entity,
                project=args.project,
                name=args.name or f"{args.project} View",
                key_panels_per_section=max(args.key_panels_per_section, 1),
                overwrite=bool(args.overwrite),
                select_run_id=select_run_id,
            )
        )
        if url:
            print(f"Workspace saved: {url}")
        else:
            print("Workspace saved.")
        return 0
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Save failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
