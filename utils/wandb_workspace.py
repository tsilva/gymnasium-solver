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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Any, Dict

# TODO: REFACTOR this file

@dataclass
class WorkspaceRequest:
    entity: str
    project: str
    name: str
    key_panels_per_section: int = 12
    overwrite: bool = False
    dry_run: bool = False
    # When provided, panels will default-select only this run via a runset filter.
    select_run_id: Optional[str] = None


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


def _maybe_build_runset(wr, *, entity: str, project: str, run_id: Optional[str]):
    """Attempt to build a Runset limited to a single run id.

    Returns the runset object or None when unsupported or run_id is None.
    """
    if not run_id:
        return None
    # Many versions expose Runset under the v2 interface; try common signatures conservatively
    try:
        # Heuristic: prefer an explicit constructor with entity/project and filters by run ids
        return wr.Runset(entity=entity, project=project, name="Latest Run", filters={"ids": [run_id]})
    except Exception:
        pass
    try:
        # Fallback: alternate arg order or parameter name
        return wr.Runset(entity, project, name="Latest Run", run_ids=[run_id])
    except Exception:
        pass
    try:
        # Fallback: query dict with id equals
        return wr.Runset(entity=entity, project=project, name="Latest Run", query={"id": {"$in": [run_id]}})
    except Exception:
        return None


def _make_panel(panel_cls, *, title: str, x: Optional[str] = None, y: Optional[Sequence[str]] = None, runset=None, range_y: Optional[Tuple[Optional[float], Optional[float]]] = None):
    """Create a panel with best-effort kwargs based on the panel signature.

    - Adds `runsets=[runset]` when supported and provided.
    - Adds `x`, `y` when supported and provided.
    - Adds `range_y=(min, max)` when supported and provided.
    Silently drops unsupported kwargs to keep compatibility across versions.
    """
    import inspect

    params = set(getattr(inspect.signature(panel_cls), "parameters", {}).keys())
    kwargs = {}
    if "title" in params:
        kwargs["title"] = title
    if x is not None and "x" in params:
        kwargs["x"] = x
    if y is not None and "y" in params:
        kwargs["y"] = list(y)
    if range_y is not None and "range_y" in params:
        kwargs["range_y"] = range_y
    if runset is not None and "runsets" in params:
        kwargs["runsets"] = [runset]
    return panel_cls(**kwargs)


def _build_sections(key_metrics: Optional[Sequence[str]], *, key_panels_per_section: int, entity: str, project: str, select_run_id: Optional[str]):
    """Compose workspace sections using wandb-workspaces constructs.

    Note: We intentionally avoid creating multiple "Key Metrics" accordions. Instead,
    metrics are grouped under their original namespaces (e.g., a single 'train'
    accordion and a single 'val' accordion) to keep the dashboard tidy.
    """
    # Import locally to keep this module import-safe when deps are missing during tests
    import wandb_workspaces.reports.v2.interface as wr
    from wandb_workspaces import workspaces as ws

    # Panels should align to the same step metric Lightning/W&B use for this
    # project. Runs log with `define_metric("*", step_metric=...)`
    # (see BaseAgent._build_trainer_loggers__wandb). Using the same key here ensures
    # workspace charts immediately render instead of showing "No metrics yet" when
    # the default "Step" column is missing from history tables.
    from utils.metrics_config import metrics_config  # type: ignore
    default_x = metrics_config.total_timesteps_key()
    sections = []
    runset = _maybe_build_runset(wr, entity=entity, project=project, run_id=select_run_id)

    # Best-effort cache to avoid repeated API/spec lookups
    _cached_action_mean_bounds: Optional[Tuple[float, float]] = None

    def _infer_action_mean_bounds_from_spec() -> Optional[Tuple[float, float]]:
        """Infer [min, max] bounds for roll/actions/mean from the env spec.

        Strategy:
        - Query W&B for the selected run (or latest in project) to obtain
          config fields (env_id, project_id).
        - Resolve spec file path using project_id when present, otherwise
          env_id normalized (ALE/Pong-v5 -> ALE-Pong-v5).
        - Read action_space from spec and derive discrete/continuous bounds.

        Falls back to None on any error so callers can omit range_y.
        """
        try:
            import wandb  # local import; optional dep
        except Exception:
            return None

        try:
            api = wandb.Api()
            run = None
            if select_run_id:
                try:
                    run = api.run(f"{entity}/{project}/{select_run_id}")
                except Exception:
                    run = None
            if run is None:
                runs = list(api.runs(f"{entity}/{project}"))
                if not runs:
                    return None
                # Most recent by created_at when available
                try:
                    runs.sort(key=lambda r: getattr(r, "created_at", None) or 0, reverse=True)
                except Exception:
                    pass
                run = runs[0]

            cfg: Dict[str, Any] = {}
            try:
                cfg = dict(getattr(run, "config", {}) or {})
            except Exception:
                cfg = {}

            project_id = str(cfg.get("project_id") or "")
            env_id = str(cfg.get("env_id") or "")
            if not project_id and not env_id:
                return None

            spec = cfg.get("spec") if isinstance(cfg.get("spec"), dict) else None
            if not isinstance(spec, dict):
                spec = _load_config_spec(project_id, env_id, cfg.get("obs_type")) or {}

            action_space = spec.get("action_space") if isinstance(spec, dict) else None
            if not isinstance(action_space, dict):
                return None

            # Prefer explicit discrete count
            if "discrete" in action_space:
                try:
                    n = int(action_space.get("discrete"))
                    if n > 0:
                        return (0.0, float(max(0, n - 1)))
                except Exception:
                    pass

            # Fallback: labels mapping 0..N-1
            labels = action_space.get("labels") if isinstance(action_space, dict) else None
            if isinstance(labels, dict) and labels:
                try:
                    keys = [int(k) for k in labels.keys()]
                    lo, hi = min(keys), max(keys)
                    return (float(lo), float(hi))
                except Exception:
                    pass

            # Continuous-style: try range/bounds
            rng = action_space.get("range") if isinstance(action_space, dict) else None
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                try:
                    lo, hi = float(rng[0]), float(rng[1])
                    return (lo, hi)
                except Exception:
                    pass
            bounds = action_space.get("bounds") if isinstance(action_space, dict) else None
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                try:
                    lo, hi = float(bounds[0]), float(bounds[1])
                    return (lo, hi)
                except Exception:
                    pass

            return None
        except Exception:
            return None

    def _y_range_for_metrics(keys: Sequence[str]) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Infer a combined y-range (min, max) from metrics.yaml for the given metric keys.

        - Uses per-metric bounds (min/max) when available in metrics.yaml.
        - For multiple series, combines mins as min(all mins) and maxes as max(all maxes).
        - Returns (None, None) when no bounds are specified for any series so the caller can omit.
        """
        # Special-case: action mean should reflect the env action space bounds
        contains_action_mean = any(str(k).endswith("roll/actions/mean") for k in keys)
        if contains_action_mean:
            nonlocal _cached_action_mean_bounds
            if _cached_action_mean_bounds is None:
                _cached_action_mean_bounds = _infer_action_mean_bounds_from_spec()
            if _cached_action_mean_bounds is not None:
                return _cached_action_mean_bounds

        mins: List[float] = []
        maxs: List[float] = []
        for k in keys:
            try:
                bounds = metrics_config.bounds_for_metric(k) or {}
            except Exception:
                bounds = {}
            if isinstance(bounds, dict):
                if "min" in bounds and bounds.get("min") is not None:
                    try:
                        mins.append(float(bounds["min"]))
                    except Exception:
                        pass
                if "max" in bounds and bounds.get("max") is not None:
                    try:
                        maxs.append(float(bounds["max"]))
                    except Exception:
                        pass
        y_min: Optional[float] = min(mins) if mins else None
        y_max: Optional[float] = max(maxs) if maxs else None
        if y_min is None and y_max is None:
            return None
        return (y_min, y_max)

    if key_metrics:
        # Treat key_priority as unnamespaced subkeys and expand across
        # common namespaces so ordering is preserved per-section.
        # If a key is already properly namespaced (e.g., "train/foo"),
        # normalize it to the target namespace by reusing its subkey only.
        def _expand_for_namespace(ns: str, keys: Sequence[str]) -> list[str]:
            expanded: list[str] = []
            seen: set[str] = set()
            for k in keys:
                if not isinstance(k, str):
                    continue
                # If already a valid namespaced metric, strip namespace
                # and reapply the target namespace to use the same order
                # across sections.
                subkey = k
                try:
                    if metrics_config.is_namespaced_metric(k):
                        _, subkey = k.split("/", 1)
                except Exception:
                    subkey = k
                full = f"{ns}/{subkey}"
                if full not in seen:
                    expanded.append(full)
                    seen.add(full)
            return expanded

        train_keys = _expand_for_namespace("train", key_metrics)
        val_keys = _expand_for_namespace("val", key_metrics)

        if train_keys:
            train_panels = []
            for k in train_keys:
                ry = _y_range_for_metrics([k])
                train_panels.append(_make_panel(wr.LinePlot, title=k, x=default_x, y=[k], runset=runset, range_y=ry))
            sections.append(ws.Section(name="train", is_open=True, panels=train_panels))

        if val_keys:
            val_panels = []
            for k in val_keys:
                ry = _y_range_for_metrics([k])
                val_panels.append(_make_panel(wr.LinePlot, title=k, x=default_x, y=[k], runset=runset, range_y=ry))
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
                        runset=runset,
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
                        runset=runset,
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
                        runset=runset,
                        range_y=_y_range_for_metrics([
                            "train/opt/loss/value_scaled",
                            "train/opt/loss/entropy_scaled",
                        ]),
                    ),
                    # ScalarChart doesn't require x/y but may also accept runsets; attempt to attach
                    _make_panel(wr.ScalarChart, title="Best Eval Reward", runset=runset),
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
    except Exception as e:
        raise ImportError("wandb-workspaces is required. pip install wandb-workspaces") from e

    # Determine repo root to source metrics config
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


def create_or_update_workspace_for_current_run(*, name: Optional[str] = None, overwrite: bool = True, key_panels_per_section: int = 12, select_current_run_only: bool = True) -> Optional[str]:
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

    # Optionally scope default selection to only the current run id
    select_run_id = getattr(run, "id", None) if (select_current_run_only and run is not None) else None

    url = create_or_update_workspace(
        WorkspaceRequest(
            entity=entity,
            project=project,
            name=name,
            overwrite=overwrite,
            key_panels_per_section=key_panels_per_section,
            select_run_id=select_run_id,
        )
    )
    return url or None
def _load_config_spec(project_id: str, env_id: str, obs_type: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort load of spec data from config/environments YAML files."""
    from utils.io import read_yaml  # local import to avoid circulars during tests

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config" / "environments"

    candidates: List[Path] = []
    if isinstance(project_id, str) and project_id:
        candidates.append(config_dir / f"{project_id}.yaml")
    if isinstance(env_id, str) and env_id:
        normalized = env_id.replace("/", "-")
        candidates.append(config_dir / f"{normalized}.yaml")

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            doc = read_yaml(path) or {}
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        if isinstance(obs_type, str):
            for key, value in doc.items():
                if isinstance(key, str) and key.startswith("_"):
                    continue
                if not isinstance(value, dict):
                    continue
                candidate = value.get("spec")
                if isinstance(candidate, dict) and value.get("obs_type") == obs_type:
                    return dict(candidate)
        spec_section = doc.get("spec")
        if isinstance(spec_section, dict):
            return dict(spec_section)
    return None
