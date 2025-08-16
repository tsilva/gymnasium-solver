"""
Publish a training run's artifacts (config, checkpoints, logs, metrics, videos) to Hugging Face Hub.

Usage:
    python publish.py --run-id <run_id> [--repo <org_or_user/repo-name>] [--private]

Behavior:
    - If --run-id is omitted, the latest run (runs/latest-run) is used.
    - If --repo is omitted, we publish to a repo named after the run's config id
        under your user/org namespace: <owner>/<config_id>.
    - Checkpoints and videos are uploaded under artifacts/, and up to 3 videos are attached
        at the repo root (preview_*.mp4) and embedded in the README for live preview on the Hub.

Authentication:
    - Ensure you are logged in: from a shell, run `huggingface-cli login` once, or set HF_TOKEN env var.

This script creates (or reuses) a model repo on the HF Hub and pushes the run directory contents
under an "artifacts/" folder preserving structure. It also generates a README model card summarizing the run
and attaches available mp4 videos as repo assets so they appear as preview media.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List
import importlib
import yaml


RUNS_DIR = Path("runs")


def resolve_run_dir(run_id: Optional[str]) -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Runs directory not found at {RUNS_DIR.resolve()}")

    if not run_id:
        # Use latest-run symlink/dir
        latest = RUNS_DIR / "latest-run"
        if not latest.exists():
            raise FileNotFoundError("No --run-id provided and 'runs/latest-run' not found")
        # If symlink, resolve target directory under runs/
        try:
            target = latest.resolve()
        except Exception:
            target = latest
        run_dir = target
        # In some setups, latest-run is a relative symlink (e.g., to <id>)
        if not run_dir.exists():
            # Fallback: iterate for most recent directory
            candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name != "latest-run"]
            if not candidates:
                raise FileNotFoundError("No runs found in runs/")
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _detect_config_id_from_run(run_dir: Path, cfg: dict) -> Optional[str]:
    """Best-effort extraction of a human-friendly config identifier used to run training.

    Priority:
      1) configs/config_id.txt (if present, single line ID)
      2) configs/config.json -> field 'config_id' (if present)
      3) Fallback to f"{env_id}_{algo_id}" when both are available
    """
    # 1) explicit text file written by training pipeline (optional)
    cfg_id_path = run_dir / "configs" / "config_id.txt"
    try:
        if cfg_id_path.exists():
            text = cfg_id_path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass

    # 2) embedded in config.json (optional)
    try:
        v = cfg.get("config_id")
        if isinstance(v, str) and v.strip():
            return v.strip()
    except Exception:
        pass

    # 3) fallback heuristic: env_id + '_' + algo_id
    env_id = cfg.get("env_id")
    algo_id = cfg.get("algo_id")
    if isinstance(env_id, str) and isinstance(algo_id, str) and env_id and algo_id:
        # Try to refine using repository environment configs to find exact config key
        guessed = _guess_config_id_from_environments(env_id, algo_id)
        return guessed or f"{env_id}_{algo_id}"


def extract_run_metadata(run_dir: Path) -> dict:
    config_path = run_dir / "configs" / "config.json"
    cfg = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except Exception:
            pass

    # Try to detect a config_id to drive the repo naming
    config_id = _detect_config_id_from_run(run_dir, cfg or {})

    # Try to find a representative checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_file = None
    if ckpt_dir.exists():
        preferred = ["best_checkpoint.ckpt", "last_checkpoint.ckpt"]
        for name in preferred:
            p = ckpt_dir / name
            if p.exists():
                ckpt_file = p
                break
        if ckpt_file is None:
            # any .ckpt
            files = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                ckpt_file = files[0]

    # Collect video previews
    videos_dir = run_dir / "videos"
    video_files: List[Path] = []
    if videos_dir.exists():
        video_files = sorted(videos_dir.rglob("*.mp4"))

    logs_dir = run_dir / "logs"
    log_files: List[Path] = []
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"))

    # Basic metrics from checkpoint (best/current eval reward, epoch, timesteps)
    metrics = {}
    if ckpt_file is not None:
        try:
            import torch  # optional; skip if not available
            data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            for k in ("best_eval_reward", "current_eval_reward", "epoch", "total_timesteps"):
                if k in data:
                    metrics[k] = float(data[k]) if isinstance(data[k], (int, float)) else data[k]
        except Exception:
            pass

    return {
        "config": cfg,
    "config_id": config_id,
        "representative_ckpt": str(ckpt_file) if ckpt_file else None,
        "videos": [str(v) for v in video_files],
        "logs": [str(l) for l in log_files],
        "metrics": metrics,
    }


def _guess_config_id_from_environments(env_id: str, algo_id: str) -> Optional[str]:
    """Search config/environments/*.yaml for a key matching env_id+algo_id.

    Preference is given to keys not starting with '__'. If multiple matches
    exist, return the first in sorted order for determinism.
    """
    try:
        root = Path(__file__).parent
        env_dir = root / "config" / "environments"
        candidates: List[str] = []
        if env_dir.exists():
            for yf in sorted(env_dir.glob("*.yaml")):
                try:
                    data = yaml.safe_load(yf.read_text()) or {}
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                for k, v in data.items():
                    if not isinstance(v, dict):
                        continue
                    if v.get("env_id") == env_id and v.get("algo_id") == algo_id:
                        candidates.append(str(k))
        # Prefer non-hidden keys
        visible = [c for c in candidates if not c.startswith("__")]
        pool = visible or candidates
        if pool:
            return sorted(pool)[0]
    except Exception:
        pass
    return None


def infer_repo_name(meta: dict, run_dir: Path, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    # Prefer the run's config_id for the repo name
    base_name = meta.get("config_id")
    if not base_name:
        # Fallback: <env>_<algo>
        algo = meta.get("config", {}).get("algo_id") or "algo"
        env_id = meta.get("config", {}).get("env_id") or "env"
        base_name = f"{env_id}_{algo}"

    user = None
    try:
        hub = importlib.import_module("huggingface_hub")
        me = hub.whoami()
        user = me.get("name")
        if not user:
            orgs = me.get("orgs", [])
            if isinstance(orgs, list) and orgs:
                first = orgs[0]
                if isinstance(first, dict):
                    user = first.get("name") or first.get("orgId")
                else:
                    user = str(first)
    except Exception:
        user = None
    # Sanitize: HF repo names allow letters, digits, - and _; avoid slashes/spaces
    base = str(base_name).strip().replace("/", "-").replace(" ", "_")
    return f"{user}/{base}" if user else base


def build_model_card(meta: dict, run_dir: Path) -> str:
    cfg = meta.get("config", {})
    algo = cfg.get("algo_id", "unknown")
    env_id = cfg.get("env_id", "unknown")
    run_id = run_dir.name
    config_id = meta.get("config_id") or f"{env_id}_{algo}"
    lines = []
    lines.append(f"# {config_id}")
    lines.append("")
    lines.append(f"Run: `{run_id}` — Env: `{env_id}` — Algo: `{algo}`")
    lines.append("")
    lines.append("This repository contains artifacts from a Gymnasium Solver training run.")
    lines.append("")
    lines.append("## Contents")
    lines.append("- Config: `artifacts/configs/config.json`")
    lines.append("- Checkpoints: `artifacts/checkpoints/*.ckpt`")
    lines.append("- Logs: `artifacts/logs/*.log`")
    if meta.get("videos"):
        lines.append("- Videos: `artifacts/videos/**/*.mp4` (also previewed below)")
    lines.append("")
    if meta.get("videos"):
        lines.append("## Previews")
        n = min(3, len(meta["videos"]))
        if n:
            for i in range(n):
                lines.append(f'<video controls src="preview_{i+1}.mp4" width="480"></video>')
            lines.append("")

    if cfg:
        lines.append("## Config (excerpt)")
        # Show a compact excerpt
        keys = [
            "env_id","algo_id","n_steps","batch_size","n_epochs","n_timesteps","seed",
            "n_envs","obs_type","policy","learning_rate","gamma","gae_lambda",
            "ent_coef","vf_coef","clip_range","normalize_advantages"
        ]
        excerpt = {k: cfg.get(k) for k in keys if k in cfg}
        lines.append("```json")
        lines.append(json.dumps(excerpt, indent=2))
        lines.append("```")
    return "\n".join(lines)


def publish_run(
    run_id: Optional[str],
    repo_id: Optional[str],
    private: bool = False,
    allow_create: bool = True,
) -> str:
    run_dir = resolve_run_dir(run_id)
    meta = extract_run_metadata(run_dir)
    card = build_model_card(meta, run_dir)

    # Make sure user is authenticated or token is set
    try:
        hub = importlib.import_module("huggingface_hub")
        HfFolder = hub.HfFolder
    except Exception as e:
        raise RuntimeError("huggingface_hub is required. Please install it (e.g., `pip install huggingface_hub`).") from e

    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN.")
    try:
        HfApi = hub.HfApi
        create_repo = hub.create_repo
        upload_file = hub.upload_file
        upload_folder = hub.upload_folder
    except Exception as e:
        raise RuntimeError("Failed to import huggingface_hub functions. Please ensure it's installed.") from e

    api = HfApi()

    # Determine repo_id
    final_repo_id = infer_repo_name(meta, run_dir, repo_id)

    # Create repo if needed
    try:
        create_repo(final_repo_id, exist_ok=True, private=private, repo_type="model")
    except Exception as e:
        if not allow_create:
            raise
        # If creation fails but repo exists, continue
        pass

    # Upload README as model card
    upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=final_repo_id,
        repo_type="model",
    )

    # Upload run artifacts preserving structure under artifacts/
    upload_folder(
        folder_path=str(run_dir),
        path_in_repo="artifacts",
        repo_id=final_repo_id,
        repo_type="model",
        allow_patterns=[
            "configs/**",
            "checkpoints/**",
            "logs/**",
            "videos/**",
            "hyperparam_control/**",
            "*.md",
        ],
        ignore_patterns=["**/__pycache__/**"],
    )

    if (run_dir / "videos").exists():

        # Also attach a few mp4s at repo root for preview
        videos = sorted((run_dir / "videos").rglob("*.mp4"))
        for i, vpath in enumerate(videos[:3]):  # Limit to first 3 previews
            upload_file(
                path_or_fileobj=str(vpath),
                path_in_repo=f"preview_{i+1}.mp4",
                repo_id=final_repo_id,
                repo_type="model",
            )

    # Save a small run-info.json
    run_info = {
        "run_id": run_dir.name,
        "meta": meta,
    }
    upload_file(
        path_or_fileobj=json.dumps(run_info, indent=2).encode("utf-8"),
        path_in_repo="artifacts/run-info.json",
        repo_id=final_repo_id,
        repo_type="model",
    )

    # Optionally upload a compact metrics.json
    if meta.get("metrics"):
        upload_file(
            path_or_fileobj=json.dumps(meta["metrics"], indent=2).encode("utf-8"),
            path_in_repo="artifacts/metrics.json",
            repo_id=final_repo_id,
            repo_type="model",
        )

    print(f"Published run {run_dir.name} to https://huggingface.co/{final_repo_id}")
    return final_repo_id


def main():
    parser = argparse.ArgumentParser(description="Publish a training run to Hugging Face Hub")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID under runs/ (defaults to latest-run)")
    parser.add_argument("--repo", type=str, default=None, help="Target repo id (e.g. user/repo). If omitted, inferred.")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    args = parser.parse_args()

    publish_run(args.run_id, args.repo, private=args.private)


if __name__ == "__main__":
    main()
