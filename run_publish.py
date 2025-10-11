"""Publish a run's artifacts to the Hugging Face Hub with a simple model card and preview video."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from utils.io import read_json, read_yaml

RUNS_DIR = Path("runs")


def resolve_run_dir(run_id: Optional[str]) -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Runs directory not found at {RUNS_DIR.resolve()}")

    if not run_id:
        # Prefer the most recently modified real run directory with artifacts,
        # but keep supporting the @last symlink (or legacy latest-run) if it points to a valid dir.
        latest = RUNS_DIR / "@last"
        latest_target: Optional[Path] = None
        if latest.exists():
            try:
                latest_target = latest.resolve()
            except OSError:
                latest_target = latest
        if latest_target is None:
            # Fallback to legacy name
            legacy = RUNS_DIR / "latest-run"
            if legacy.exists():
                try:
                    latest_target = legacy.resolve()
                except OSError:
                    latest_target = legacy

        # Collect candidate run dirs (exclude the latest-run entries themselves)
        candidates = [
            p for p in RUNS_DIR.iterdir()
            if p.is_dir() and p.name not in {"@last", "latest-run"}
        ]
        if not candidates and not latest_target:
            raise FileNotFoundError("No runs found in runs/")

        def has_artifacts(p: Path) -> Tuple[int, bool]:
            """Score by mtime and presence of interesting files (checkpoints)."""
            mtime = int(p.stat().st_mtime)
            ckpts = any(p.glob("checkpoints/*.ckpt")) or any(p.glob("checkpoints/**/*.ckpt"))
            return (mtime, ckpts)

        # Choose best candidate: prefer one with artifacts; tie-breaker by mtime
        best = None
        if candidates:
            best = max(candidates, key=lambda p: (has_artifacts(p)[1], has_artifacts(p)[0]))

        run_dir = None
        # If latest-target exists and has artifacts, prefer it; otherwise use best
        if latest_target and latest_target.exists():
            lt_score = has_artifacts(latest_target)
            if best is None or lt_score >= has_artifacts(best):
                run_dir = latest_target
        if run_dir is None:
            run_dir = best or latest_target  # fallback to whatever exists
        if run_dir is None:
            raise FileNotFoundError("Unable to resolve a valid run directory")
    else:
        # Resolve @last symlink if needed
        if run_id == "@last":
            latest = RUNS_DIR / "@last"
            if not latest.exists():
                raise FileNotFoundError("No @last run found. Train a model first.")
            run_id = latest.resolve().name

        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            # Try to download from W&B
            print(f"Run {run_id} not found locally. Attempting to download from W&B...")
            from utils.wandb_artifacts import download_run_artifact
            download_run_artifact(run_id)
            # Verify it exists now
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _detect_config_id_from_run(run_dir: Path, cfg: dict) -> Optional[str]:
    """Infer a human-friendly config_id from run files or env+algo fallback."""
    # 1) explicit text file written by training pipeline (optional)
    cfg_id_path = run_dir / "configs" / "config_id.txt"
    if cfg_id_path.exists():
        text = cfg_id_path.read_text(encoding="utf-8").strip()
        if text:
            return text

    # 2) embedded in config.json (optional)
    v = cfg.get("config_id")
    if isinstance(v, str) and v.strip():
        return v.strip()

    # 3) fallback heuristic: env_id + '_' + algo_id
    env_id = cfg.get("env_id")
    algo_id = cfg.get("algo_id")
    if isinstance(env_id, str) and isinstance(algo_id, str) and env_id and algo_id:
        # Try to refine using repository environment configs to find exact config key
        guessed = _guess_config_id_from_environments(env_id, algo_id)
        return guessed or f"{env_id}_{algo_id}"


def _find_videos_for_run(run_dir: Path) -> List[Path]:
    """Find run videos under runs/<id>/checkpoints or mirrored W&B paths."""
    videos: List[Path] = []
    local_checkpoints_root = run_dir / "checkpoints"
    if local_checkpoints_root.exists():
        videos = sorted(local_checkpoints_root.rglob("*.mp4"))
        if videos:
            return videos

    # Fallback: try within wandb run files mirrors
    repo_root = Path(__file__).parent
    wandb_root = repo_root / "wandb"
    if wandb_root.exists():
        run_id = run_dir.name
        pattern = f"run-*/files/runs/{run_id}/checkpoints"
        for path in sorted(wandb_root.glob(pattern)):
            if path.is_dir():
                vids = sorted(Path(path).rglob("*.mp4"))
                videos.extend(vids)
        # Also check wandb/@last link if present (fallback to legacy)
        latest = wandb_root / "@last" / "files" / "runs" / run_id / "checkpoints"
        if latest.exists():
            videos.extend(sorted(latest.rglob("*.mp4")))
        else:
            legacy = wandb_root / "latest-run" / "files" / "runs" / run_id / "checkpoints"
            if legacy.exists():
                videos.extend(sorted(legacy.rglob("*.mp4")))
    # Return unique, ordered
    seen = set()
    uniq: List[Path] = []
    for v in videos:
        try:
            rp = v.resolve()
        except OSError:
            rp = v
        if rp not in seen:
            uniq.append(Path(rp))
            seen.add(rp)
    return uniq


def _find_best_video_for_run(run_dir: Path) -> Optional[Path]:
    """Locate the canonical best evaluation video for the run."""
    # Videos are now stored in checkpoints directories
    # Look for best checkpoint video first (inside @best symlink)
    best_checkpoint = run_dir / "checkpoints" / "@best"
    if best_checkpoint.exists():
        for candidate in ["best.mp4", "final.mp4", "best_checkpoint.mp4"]:
            video_path = best_checkpoint / candidate
            if video_path.exists():
                return video_path.resolve()

    # Fallback: scan all checkpoints and pick the first matching filename
    for v in _find_videos_for_run(run_dir):
        if v.name in {"best.mp4", "final.mp4", "best_checkpoint.mp4"}:
            try:
                return v.resolve()
            except OSError:
                return v
    return None


def extract_run_metadata(run_dir: Path) -> dict:
    # Prefer new layout: config.json at run root; fallback to legacy path
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path = run_dir / "configs" / "config.json"
    cfg = {}
    if config_path.exists():
        cfg = read_json(config_path) or {}

    # Try to detect a config_id to drive the repo naming
    config_id = _detect_config_id_from_run(run_dir, cfg or {})

    # Try to find a representative checkpoint
    ckpt_dir = run_dir / "checkpoints"
    ckpt_file = None
    if ckpt_dir.exists():
        preferred = ["best.ckpt", "last.ckpt", "best.ckpt", "last.ckpt"]
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

    # Find the best video (best.mp4 or legacy best_checkpoint.mp4) if available
    best_video: Optional[Path] = _find_best_video_for_run(run_dir)

    # Logs: prefer stable run.log at run root, then all logs under logs/
    logs_dir = run_dir / "logs"
    log_files: List[Path] = []
    stable_log = run_dir / "run.log"
    if stable_log.exists():
        log_files.append(stable_log)
    if logs_dir.exists():
        log_files.extend(sorted(logs_dir.glob("*.log")))

    # Basic metrics from checkpoint (best/current eval reward, epoch, timesteps)
    metrics = {}
    if ckpt_file is not None:
        import torch  # optional; skip if not available
        data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        for k in ("best_eval_reward", "current_eval_reward", "epoch", "total_timesteps"):
            if k in data:
                metrics[k] = float(data[k]) if isinstance(data[k], (int, float)) else data[k]

    return {
        "config": cfg,
    "config_id": config_id,
        "representative_ckpt": str(ckpt_file) if ckpt_file else None,
    "best_video": str(best_video) if best_video else None,
        "logs": [str(l) for l in log_files],
        "metrics": metrics,
    }


def _guess_config_id_from_environments(env_id: str, algo_id: str) -> Optional[str]:
    """Search config/environments for a config_id matching env_id+algo_id (legacy and new styles)."""
    root = Path(__file__).parent
    env_dir = root / "config" / "environments"
    candidates: List[str] = []
    if env_dir.exists():
        for yf in sorted(env_dir.glob("*.yaml")):
            if yf.name.endswith(".new.yaml"):
                continue
            doc = read_yaml(yf) or {}
            if not isinstance(doc, dict):
                continue
            # Detect new style (support `_base` anchor mapping as well)
            project = (
                doc.get("project_id")
                or (doc.get("_base") or {}).get("project_id")
                or yf.stem.replace(".new", "")
            )
            has_env_root = isinstance(doc.get("env_id"), str)
            has_base = isinstance(doc.get("_base"), dict)
            base_env = None
            if has_env_root:
                base_env = doc.get("env_id")
            elif has_base and isinstance(doc["_base"].get("env_id"), str):
                base_env = doc["_base"].get("env_id")
            if base_env is not None:
                # base at root or under _base, variants under keys that are dicts
                for k, v in doc.items():
                    if not isinstance(v, dict):
                        continue
                    # Skip meta/utility sections (e.g., _base)
                    if isinstance(k, str) and k.startswith("_"):
                        continue
                    # algo_id may be inside the variant, else assume section name
                    cand_algo = v.get("algo_id") or str(k)
                    if base_env == env_id and cand_algo == algo_id:
                        candidates.append(f"{project}_{k}")
            else:
                # Legacy style: id -> mapping
                for k, v in doc.items():
                    if not isinstance(v, dict):
                        continue
                    if v.get("env_id") == env_id and v.get("algo_id") == algo_id:
                        candidates.append(str(k))
    # Prefer non-hidden keys
    visible = [c for c in candidates if not c.startswith("__")]
    pool = visible or candidates
    if pool:
        return sorted(pool)[0]
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
    # Sanitize: HF repo names allow letters, digits, - and _; avoid slashes/spaces
    base = str(base_name).strip().replace("/", "-").replace(" ", "_")
    return f"{user}/{base}" if user else base


def build_model_card(meta: dict, run_dir: Path) -> str:
    """Build a minimal README model card with YAML front matter and optional preview video."""
    cfg = meta.get("config", {})
    algo = cfg.get("algo_id", "unknown")
    env_id = cfg.get("env_id", "unknown")
    run_id = run_dir.name
    config_id = meta.get("config_id") or f"{env_id}_{algo}"

    # Compose YAML front matter (keep it simple but non-empty)
    tags = [
        "reinforcement-learning",
        "gymnasium",
        str(env_id) if env_id else "env",
        str(algo) if algo else "algo",
        "pytorch",
    ]

    # Optional metrics for model-index
    metrics_yaml = []
    m = meta.get("metrics") or {}
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    if isinstance(m, dict):
        if _num(m.get("best_eval_reward")) is not None:
            metrics_yaml.append({"name": "Best Eval Reward", "type": "reward", "value": _num(m.get("best_eval_reward"))})
        if _num(m.get("current_eval_reward")) is not None:
            metrics_yaml.append({"name": "Current Eval Reward", "type": "reward", "value": _num(m.get("current_eval_reward"))})
        if _num(m.get("epoch")) is not None:
            metrics_yaml.append({"name": "Epoch", "type": "epoch", "value": _num(m.get("epoch"))})
        if _num(m.get("total_timesteps")) is not None:
            metrics_yaml.append({"name": "Total Timesteps", "type": "timesteps", "value": _num(m.get("total_timesteps"))})

    # Build front matter
    front_lines: List[str] = []
    front_lines.append("---")
    front_lines.append("license: mit")
    front_lines.append("library_name: pytorch")
    # Make the task explicit so the Hub enables the RL preview panel
    front_lines.append("pipeline_tag: reinforcement-learning")
    # optional language for better discoverability
    front_lines.append("language:")
    front_lines.append("  - en")
    front_lines.append("tags:")
    for t in tags:
        front_lines.append(f"  - {t}")
    # Minimal model-index if we have any metrics
    if metrics_yaml:
        front_lines.append("model-index:")
        front_lines.append("  - name: " + str(config_id))
        front_lines.append("    results:")
        front_lines.append("      - task:")
        front_lines.append("          type: reinforcement-learning")
        front_lines.append("          name: Reinforcement Learning")
        front_lines.append("        dataset:")
        front_lines.append("          name: " + str(env_id))
        front_lines.append("          type: gymnasium")
        front_lines.append("        metrics:")
        for item in metrics_yaml:
            # Each metric is a simple flat mapping
            front_lines.append("          - name: " + str(item["name"]))
            front_lines.append("            type: " + str(item["type"]))
            front_lines.append("            value: " + str(item["value"]))
    front_lines.append("---")

    lines = []
    lines.extend(front_lines)
    lines.append("")
    lines.append(f"# {config_id}")
    lines.append("")
    lines.append(f"Run: `{run_id}` — Env: `{env_id}` — Algo: `{algo}`")
    lines.append("")
    lines.append("This repository contains artifacts from a Gymnasium Solver training run.")
    lines.append("")
    lines.append("## Contents")
    lines.append("- Config: `artifacts/config.json`")
    lines.append("- Checkpoints: `artifacts/checkpoints/*.ckpt`")
    lines.append("- Logs: `artifacts/logs/*.log`")
    if meta.get("best_video"):
        lines.append("- Video: `artifacts/checkpoints/**/best.mp4` (also previewed below)")
    lines.append("")
    if meta.get("best_video"):
        lines.append("## Preview")
        # Prefer preview.mp4 but also provide a fallback link to replay.mp4
        lines.append('<video controls src="preview.mp4" width="480"></video>')
        lines.append("")
        lines.append("If the video above doesn't load, try the fallback: [replay.mp4](replay.mp4)")
        lines.append("")

    if cfg:
        lines.append("## Config (excerpt)")
        # Show a compact excerpt
        keys = [
            "env_id","algo_id","n_steps","batch_size","n_epochs","max_timesteps","seed",
            "n_envs","obs_type","policy","policy_lr","gamma","gae_lambda",
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
) -> str:
    run_dir = resolve_run_dir(run_id)
    meta = extract_run_metadata(run_dir)
    card = build_model_card(meta, run_dir)

    # Make sure user is authenticated or token is set
    hub = importlib.import_module("huggingface_hub")
    HfFolder = hub.HfFolder

    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run `huggingface-cli login` or set HF_TOKEN.")
    create_repo = hub.create_repo
    upload_file = hub.upload_file
    upload_folder = hub.upload_folder

    # Determine repo_id
    final_repo_id = infer_repo_name(meta, run_dir, repo_id)

    # Create repo if needed
    create_repo(final_repo_id, exist_ok=True, private=private, repo_type="model")

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
            "checkpoints/**",  # includes videos stored inside checkpoint directories
            "logs/**",
            "hyperparam_control/**",
            "*.md",
        ],
        ignore_patterns=["**/__pycache__/**"],
    )

    # Also attach a few mp4s at repo root for preview (from whatever sources we detected)
    best_video = meta.get("best_video")
    if best_video:
        upload_file(
            path_or_fileobj=str(best_video),
            path_in_repo="preview.mp4",
            repo_id=final_repo_id,
            repo_type="model",
        )
        # Upload under additional common names recognized by the Hub UI
        # (SB3 and others typically use replay.mp4).
        for alt_name in ("replay.mp4", "video-preview.mp4"):
            upload_file(
                path_or_fileobj=str(best_video),
                path_in_repo=alt_name,
                repo_id=final_repo_id,
                repo_type="model",
            )

    # If the best video lives outside the run_dir (e.g., W&B mirrors),
    # upload it explicitly under artifacts so it is preserved.
    if best_video:
        run_dir_resolved = run_dir.resolve()
        vpath = Path(best_video)
        rel_ok = True
        try:
            vpath.resolve().relative_to(run_dir_resolved)
        except ValueError:
            rel_ok = False
        if not rel_ok:
            upload_file(
                path_or_fileobj=str(vpath),
                path_in_repo=f"artifacts/checkpoints/_external/{vpath.name}",
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
    parser.add_argument("--run-id", type=str, default=None, help="Run ID under runs/ (defaults to @last)")
    parser.add_argument("--repo", type=str, default=None, help="Target repo id (e.g. user/repo). If omitted, inferred.")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    args = parser.parse_args()

    publish_run(args.run_id, args.repo, private=args.private)


if __name__ == "__main__":
    main()
