"""
Utilities to create or update a W&B Workspace (dashboard) for a project.

Dependencies: `wandb`, `wandb-workspaces` (declared in project deps).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

@dataclass
class WorkspaceRequest:
    entity: str
    project: str
    name: str
    key_panels_per_section: int = 12
    overwrite: bool = False
    dry_run: bool = False
    select_run_id: Optional[str] = None  # Currently unused


def _read_key_metrics(repo_root: Path) -> Optional[Sequence[str]]:
    """Read key metrics order from config/metrics.yaml."""
    metrics_yaml = repo_root / "config" / "metrics.yaml"
    if not metrics_yaml.exists():
        return None
    from utils.io import read_yaml
    cfg = read_yaml(metrics_yaml)
    global_cfg = cfg.get("_global", {})
    key_metrics = global_cfg.get("key_priority")
    return [str(k) for k in key_metrics] if key_metrics else None


def _make_panel(panel_cls, *, title: str, x: Optional[str] = None, y: Optional[Sequence[str]] = None, range_y=None):
    """Create a panel with provided parameters."""
    kwargs = {"title": title}
    if x is not None:
        kwargs["x"] = x
    if y is not None:
        kwargs["y"] = list(y)
    if range_y is not None:
        kwargs["range_y"] = range_y
    return panel_cls(**kwargs)


def _build_sections(key_metrics: Optional[Sequence[str]], *, key_panels_per_section: int, entity: str, project: str, select_run_id: Optional[str]):
    """Compose workspace sections using wandb-workspaces constructs."""
    import wandb_workspaces.reports.v2.interface as wr
    from wandb_workspaces import workspaces as ws
    from utils.metrics_config import metrics_config

    default_x = metrics_config.total_timesteps_key()
    sections = []

    def _y_range_for_metrics(keys: Sequence[str]):
        """Get combined y-range from metrics.yaml."""
        mins, maxs = [], []
        for k in keys:
            bounds = metrics_config.bounds_for_metric(k) or {}
            if "min" in bounds and bounds["min"] is not None:
                mins.append(float(bounds["min"]))
            if "max" in bounds and bounds["max"] is not None:
                maxs.append(float(bounds["max"]))
        y_min = min(mins) if mins else None
        y_max = max(maxs) if maxs else None
        return (y_min, y_max) if y_min is not None or y_max is not None else None

    if key_metrics:
        def _expand_for_namespace(ns: str, keys: Sequence[str]) -> list[str]:
            """Expand metric keys for namespace, stripping existing namespace if present."""
            expanded = []
            for k in keys:
                subkey = k.split("/", 1)[1] if metrics_config.is_namespaced_metric(k) else k
                full = f"{ns}/{subkey}"
                if full not in expanded:
                    expanded.append(full)
            return expanded

        train_keys = _expand_for_namespace("train", key_metrics)
        val_keys = _expand_for_namespace("val", key_metrics)

        if train_keys:
            train_panels = []
            for k in train_keys:
                ry = _y_range_for_metrics([k])
                train_panels.append(_make_panel(wr.LinePlot, title=k, x=default_x, y=[k], range_y=ry))
            sections.append(ws.Section(name="train", is_open=True, panels=train_panels))

        if val_keys:
            val_panels = []
            for k in val_keys:
                ry = _y_range_for_metrics([k])
                val_panels.append(_make_panel(wr.LinePlot, title=k, x=default_x, y=[k], range_y=ry))
            sections.append(ws.Section(name="val", is_open=True, panels=val_panels))

    # Diagnostics + Videos sections
    sections.extend(
        [
            ws.Section(
                name="Diagnostics",
                is_open=True,
                panels=[
                    _make_panel(
                        wr.LinePlot,
                        title="Policy LR (scheduled)",
                        x=default_x,
                        y=["train/policy_lr"],
                        range_y=_y_range_for_metrics(["train/policy_lr"]),
                    ),
                    _make_panel(
                        wr.LinePlot,
                        title="Losses",
                        x=default_x,
                        y=[
                            "train/opt/loss/policy",
                            "train/opt/loss/value",
                            "train/opt/policy/entropy",
                        ],
                        range_y=_y_range_for_metrics([
                            "train/opt/loss/policy",
                            "train/opt/loss/value",
                            "train/opt/policy/entropy",
                        ]),
                    ),
                    _make_panel(
                        wr.LinePlot,
                        title="Scaled Losses",
                        x=default_x,
                        y=[
                            "train/opt/loss/value_scaled",
                            "train/opt/loss/entropy_scaled",
                        ],
                        range_y=_y_range_for_metrics([
                            "train/opt/loss/value_scaled",
                            "train/opt/loss/entropy_scaled",
                        ]),
                    ),
                    _make_panel(wr.ScalarChart, title="Best Eval Reward"),
                ],
            ),
            ws.Section(
                name="Videos",
                is_open=False,
                panels=[
                    # MediaBrowser may not support runsets; construct without runset
                    wr.MediaBrowser(title="Recent Episodes", media_keys=["train/roll/episodes", "val/roll/episodes"])],
            ),
        ]
    )

    return sections


def create_or_update_workspace(req: WorkspaceRequest) -> str:
    """Create or update a W&B workspace for a project and return its URL."""
    assert req.entity and req.project, "Both entity and project must be provided"

    # Cleanup WANDB_BASE_URL if it ends with /graphql
    base_url = os.environ.get("WANDB_BASE_URL")
    if base_url and base_url.rstrip("/").endswith("/graphql"):
        os.environ["WANDB_BASE_URL"] = base_url.rstrip("/")[: -len("/graphql")]

    import wandb
    from wandb_workspaces import workspaces as ws
    from wandb_gql import gql

    repo_root = Path(__file__).resolve().parents[1]
    key_metrics = _read_key_metrics(repo_root)
    sections = _build_sections(
        key_metrics,
        key_panels_per_section=req.key_panels_per_section,
        entity=req.entity,
        project=req.project,
        select_run_id=req.select_run_id,
    )

    workspace = ws.Workspace(name=req.name, entity=req.entity, project=req.project, sections=sections)

    if req.dry_run:
        import json
        return json.dumps(workspace.to_json(), indent=2)

    # Ensure project exists
    api = wandb.Api()
    projects = list(api.projects(req.entity))
    if not any(p.name == req.project for p in projects):
        api.create_project(req.project, req.entity)

    # Check for existing workspace
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
    edges = resp.get("project", {}).get("allViews", {}).get("edges", [])

    internal_name = None
    for e in edges:
        node = e.get("node", {})
        if node.get("displayName") == req.name:
            internal_name = node.get("name")
            break

    # Return existing workspace URL if not overwriting
    if internal_name and not req.overwrite:
        from wandb_workspaces.workspaces import internal as ws_internal
        base = api.client.app_url.rstrip("/")
        nw = ws_internal._internal_name_to_url_query_str(internal_name)
        return f"{base}/{req.entity}/{req.project}?nw={nw}"

    # Create or update workspace
    if req.overwrite and internal_name:
        from wandb_workspaces.workspaces import internal as ws_internal
        base = api.client.app_url.rstrip("/")
        nw = ws_internal._internal_name_to_url_query_str(internal_name)
        url = f"{base}/{req.entity}/{req.project}?nw={nw}"
        existing_ws = ws.Workspace.from_url(url)
        existing_ws.sections = sections
        saved_ws = existing_ws.save()
    else:
        saved_ws = workspace.save()

    return str(saved_ws.url)


def create_or_update_workspace_for_current_run(*, name: Optional[str] = None, overwrite: bool = True, key_panels_per_section: int = 12, select_current_run_only: bool = True) -> Optional[str]:
    """Derive entity/project from active wandb run and push a workspace."""
    import wandb

    run = wandb.run
    entity = (run.entity if run else None) or os.getenv("WANDB_ENTITY")
    project = (run.project if run else None) or os.getenv("WANDB_PROJECT")

    if not entity or not project:
        return None

    name = name or f"{project} View"
    select_run_id = run.id if (select_current_run_only and run) else None

    return create_or_update_workspace(
        WorkspaceRequest(
            entity=entity,
            project=project,
            name=name,
            overwrite=overwrite,
            key_panels_per_section=key_panels_per_section,
            select_run_id=select_run_id,
        )
    )
