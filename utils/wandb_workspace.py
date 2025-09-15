"""
Utilities to create or update a W&B Workspace (dashboard) for a project.

This module extracts the logic from `scripts/setup_wandb_dashboard.py` so it can be
reused both from that script and programmatically (e.g., at end of training).

Key behavior:
- Builds a default workspace layout using key metrics from `config/metrics.yaml` when available.
- Ensures the W&B project exists before creating/updating the workspace.
- Can update (overwrite) an existing workspace by display name.
- Returns a URL to the saved workspace when possible.

Dependencies: `wandb`, `wandb-workspaces` (declared in project deps).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import os


@dataclass
class WorkspaceRequest:
    entity: str
    project: str
    name: str
    key_panels_per_section: int = 12
    overwrite: bool = False
    dry_run: bool = False


def _read_key_metrics(repo_root: Path) -> Optional[Sequence[str]]:
    """Best-effort read of key metrics order from config/metrics.yaml.

    Returns a list of metric keys or None when not available.
    """
    metrics_yaml = repo_root / "config" / "metrics.yaml"
    if not metrics_yaml.exists():
        return None
    try:
        # Prefer project IO helper if available
        try:
            from utils.io import read_yaml  # type: ignore

            cfg = read_yaml(metrics_yaml)
        except Exception:
            import yaml  # type: ignore

            with metrics_yaml.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        global_cfg = cfg.get("_global", {}) if isinstance(cfg, dict) else {}
        key_metrics = global_cfg.get("key_priority") or global_cfg.get("keypriority")
        if key_metrics is None:
            return None
        return [str(k) for k in key_metrics if k is not None]
    except Exception:
        return None


def _build_sections(key_metrics: Optional[Sequence[str]], *, key_panels_per_section: int):
    """Compose workspace sections using wandb-workspaces constructs."""
    # Import locally to keep this module import-safe when deps are missing during tests
    from wandb_workspaces import workspaces as ws
    import wandb_workspaces.reports.v2.interface as wr

    default_x = "Step"
    sections = []

    if key_metrics:
        # Partition key metrics into multiple sections to avoid overcrowding
        def _chunks(seq, size):
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        for idx, group in enumerate(_chunks(list(key_metrics), max(int(key_panels_per_section), 1)), start=1):
            panels = [wr.LinePlot(title=m, x=default_x, y=[m]) for m in group]
            name = "Key Metrics" if idx == 1 else f"Key Metrics {(idx - 1) * key_panels_per_section + 1}–{(idx - 1) * key_panels_per_section + len(group)}"
            sections.append(ws.Section(name=name, is_open=True, panels=panels))

    # Diagnostics + Videos sections
    sections.extend(
        [
            ws.Section(
                name="Diagnostics",
                is_open=True,
                panels=[
                    wr.LinePlot(title="Policy LR (scheduled)", x=default_x, y=["train/policy_lr"]),
                    wr.LinePlot(title="Losses", x=default_x, y=["train/policy_loss", "train/value_loss", "train/entropy"]),
                    wr.ScalarChart(title="Best Eval Reward", metric="val/ep_rew_best", groupby_aggfunc="max"),
                ],
            ),
            ws.Section(
                name="Videos",
                is_open=False,
                panels=[wr.MediaBrowser(title="Recent Episodes", media_keys=["train/episodes", "val/episodes"])],
            ),
        ]
    )

    return sections


def create_or_update_workspace(req: WorkspaceRequest) -> str:
    """Create or update a W&B workspace for a project and return its URL.

    This is idempotent when `overwrite=False` and will update the existing view when
    `overwrite=True`.

    Raises ImportError for missing soft dependencies, ValueError for missing args,
    and propagates other exceptions from W&B APIs.
    """
    if not req.entity or not req.project:
        raise ValueError("Both entity and project must be provided")

    # Optional cleanup: some environments set WANDB_BASE_URL ending with /graphql which breaks APIs
    base_url = os.environ.get("WANDB_BASE_URL")
    if base_url and base_url.rstrip("/").endswith("/graphql"):
        os.environ["WANDB_BASE_URL"] = base_url.rstrip("/")[: -len("/graphql")]

    try:
        import wandb
    except Exception as e:  # Narrow: optional dependency for pushing workspaces
        raise ImportError("wandb is required. pip install wandb") from e

    try:
        # Workspaces and panel primitives
        from wandb_workspaces import workspaces as ws
        import wandb_workspaces.reports.v2.interface as wr  # noqa: F401 - ensure importable
    except Exception as e:
        raise ImportError("wandb-workspaces is required. pip install wandb-workspaces") from e

    # Determine repo root to source metrics config
    repo_root = Path(__file__).resolve().parents[1]
    key_metrics = _read_key_metrics(repo_root)
    sections = _build_sections(key_metrics, key_panels_per_section=req.key_panels_per_section)

    workspace = ws.Workspace(name=req.name, entity=req.entity, project=req.project, sections=sections)

    if req.dry_run:
        try:
            # Some versions expose to_json(); fall back to a minimal dict when absent
            as_json = workspace.to_json()  # type: ignore[attr-defined]
        except Exception:
            as_json = {
                "name": req.name,
                "entity": req.entity,
                "project": req.project,
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

        return json.dumps(as_json, indent=2)

    # Ensure project exists
    api = wandb.Api()
    projects = list(api.projects(req.entity))
    if not any(getattr(p, "name", None) == req.project for p in projects):
        api.create_project(req.project, req.entity)

    # Check for an existing workspace with this display name
    existing = None
    internal_name = None
    from wandb_gql import gql

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
    resp = api.client.execute(query, {"entityName": req.entity, "projectName": req.project, "viewType": "project-view"})
    edges = (((resp or {}).get("project") or {}).get("allViews") or {}).get("edges", [])
    for e in edges:
        node = e.get("node") or {}
        if node.get("displayName") == req.name:
            existing = node
            internal_name = node.get("name")
            break

    # If not overwriting and a workspace already exists, return its URL without saving
    if existing and not req.overwrite:
        try:
            from wandb_workspaces.workspaces import internal as ws_internal

            base = wandb.Api().client.app_url.rstrip("/")
            nw = ws_internal._internal_name_to_url_query_str(internal_name or "")
            return f"{base}/{req.entity}/{req.project}?nw={nw}"
        except Exception:
            base = wandb.Api().client.app_url.rstrip("/")
            return f"{base}/{req.entity}/{req.project}"

    # Create or update
    saved_ws = None
    if req.overwrite and internal_name:
        try:
            from wandb_workspaces.workspaces import internal as ws_internal

            base = wandb.Api().client.app_url.rstrip("/")
            nw = ws_internal._internal_name_to_url_query_str(internal_name)
            url = f"{base}/{req.entity}/{req.project}?nw={nw}"
            existing_ws = ws.Workspace.from_url(url)
            existing_ws.sections = sections
            saved_ws = existing_ws.save()
        except Exception:
            saved_ws = None
    if saved_ws is None:
        saved_ws = workspace.save()

    # Provide a user-friendly URL
    try:
        return str(saved_ws.url)
    except Exception:
        # Best-effort reconstruction when .url is not present
        try:
            base = wandb.Api().client.app_url.rstrip("/")
            if internal_name:
                from wandb_workspaces.workspaces import internal as ws_internal

                nw = ws_internal._internal_name_to_url_query_str(internal_name)
                return f"{base}/{req.entity}/{req.project}?nw={nw}"
            return f"{base}/{req.entity}/{req.project}"
        except Exception:
            return ""


def create_or_update_workspace_for_current_run(*, name: Optional[str] = None, overwrite: bool = True, key_panels_per_section: int = 12) -> Optional[str]:
    """Convenience: derive entity/project from the active wandb run and push a workspace.

    Returns the workspace URL on success, None when no active run or when required
    information is missing. Exceptions from W&B APIs propagate to the caller.
    """
    try:
        import wandb
    except Exception:
        return None

    run = getattr(wandb, "run", None)
    if run is None:
        # Fallback to env vars when no active run is present
        entity = os.getenv("WANDB_ENTITY")
        project = os.getenv("WANDB_PROJECT")
    else:
        # Prefer properties from the active run
        entity = getattr(run, "entity", None) or os.getenv("WANDB_ENTITY")
        project = getattr(run, "project", None) or os.getenv("WANDB_PROJECT")

    if not entity or not project:
        return None

    if not name:
        name = f"{project} View"

    url = create_or_update_workspace(
        WorkspaceRequest(
            entity=entity,
            project=project,
            name=name,
            overwrite=overwrite,
            key_panels_per_section=key_panels_per_section,
        )
    )
    return url or None
