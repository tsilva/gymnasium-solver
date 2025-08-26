#!/usr/bin/env python3
"""
Inspector application logic (separated to avoid stdlib inspect shadowing).

Run via: python inspect.py [args]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Avoid macOS AppKit main-thread violations when environments initialize
# Pygame/SDL from Gradio worker threads. Using the headless SDL drivers
# prevents Cocoa window/menu initialization on non-main threads.
import os
import platform
if platform.system() == "Darwin":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from play import load_model as _load_model
from play import load_config_from_run as _load_config_from_run
from utils.rollouts import compute_mc_returns, compute_gae_advantages_and_returns


RUNS_DIR = Path("runs")


def _resolve_run_dir(run_id: str) -> Path:
    if run_id == "latest-run":
        latest = RUNS_DIR / "latest-run"
        if latest.is_symlink():
            run_id = str(latest.readlink())
        else:
            raise FileNotFoundError("latest-run symlink not found")
    run_path = RUNS_DIR / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    return run_path


def list_runs() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [p.name for p in RUNS_DIR.iterdir() if p.is_dir() and p.name != "latest-run"]
    runs.sort(key=lambda n: (RUNS_DIR / n).stat().st_mtime, reverse=True)
    return runs


def list_checkpoints_for_run(run_id: str) -> Tuple[List[str], Dict[str, Path], str | None]:
    run_dir = _resolve_run_dir(run_id)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return [], {}, None

    files = list(ckpt_dir.glob("*.ckpt"))
    if not files:
        return [], {}, None

    def score(p: Path):
        name = p.name
        if name in {"best.ckpt", "best.ckpt"}:  # support legacy and new names
            return (0, -p.stat().st_mtime)
        if name in {"last.ckpt", "last.ckpt"}:
            return (1, -p.stat().st_mtime)
        return (2, -p.stat().st_mtime)

    files.sort(key=score)

    labels: List[str] = []
    mapping: Dict[str, Path] = {}
    for p in files:
        label = p.name
        if label in {"best.ckpt", "best.ckpt"}:
            label = "best.ckpt (best)"
        elif label in {"last.ckpt", "last.ckpt"}:
            label = "last.ckpt (last)"
        labels.append(label)
        mapping[label] = p

    default_label = next((l for l in labels if l.startswith("best")), labels[0])
    return labels, mapping, default_label


def _load_env_info_yaml(env_id: str) -> Dict[str, Any] | None:
    """
    Load env_info YAML for the given env_id.

    Search order:
      1) ENV_INFO_DIR environment variable
      2) Project default: <project_root>/config/env_info

    Supports nested env IDs like 'ALE/Pong-v5' → 'ALE/Pong-v5.yaml'.
    """
    import yaml  # local import to avoid hard dependency when unused
    import os

    candidates: List[Path] = []
    custom_dir = os.environ.get("ENV_INFO_DIR")
    if custom_dir:
        candidates.append(Path(custom_dir))
    try:
        # inspector_app.py lives at the project root
        project_root = Path(__file__).resolve().parent
        candidates.append(project_root / "config" / "env_info")
    except Exception:
        pass

    for base in candidates:
        try:
            path = base / f"{env_id}.yaml"
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
        except Exception:
            continue
    return None


def _extract_action_labels(env_info: Dict[str, Any] | None) -> List[str] | None:
    """Return ordered list of action labels if present and valid, else None."""
    if not isinstance(env_info, dict):
        return None
    try:
        action_space = env_info.get("action_space") or {}
        discrete = action_space.get("discrete")
        labels_map = action_space.get("labels")
        if not isinstance(discrete, int) or discrete <= 0 or not isinstance(labels_map, dict):
            return None
        ordered: List[str] = []
        for i in range(discrete):
            label = labels_map.get(i)
            if not isinstance(label, str):
                return None
            ordered.append(label)
        return ordered
    except Exception:
        return None


def _to_batch_obs(obs) -> torch.Tensor:
    if isinstance(obs, np.ndarray) and obs.ndim > 1:
        return torch.as_tensor(obs, dtype=torch.float32)
    return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)


def _ensure_action_shape(env, action):
    if hasattr(env, "num_envs") and env.num_envs == 1:
        if not isinstance(action, np.ndarray):
            action = np.array([action])
    return action


def run_episode(
    run_id: str,
    checkpoint_label: str | None,
    deterministic: bool = False,
    max_steps: int = 1000,
) -> Tuple[List[np.ndarray], List[np.ndarray] | None, List[np.ndarray] | None, List[Dict[str, Any]], Dict[str, Any]]:
    config = _load_config_from_run(run_id)

    labels, mapping, default_label = list_checkpoints_for_run(run_id)
    if not labels:
        raise FileNotFoundError("No checkpoints found for this run")
    selected_label = checkpoint_label or default_label
    ckpt_path = mapping[selected_label]

    from utils.environment import build_env

    env = build_env(
        config.env_id,
        seed=42,
        env_wrappers=config.env_wrappers,
        norm_obs=config.normalize_obs,
        n_envs=1,
        frame_stack=config.frame_stack,
        obs_type=config.obs_type,
        render_mode="rgb_array",
        env_kwargs=config.env_kwargs,
        subproc=False,
        # Match training-time preprocessing so model input shapes align
        grayscale_obs=getattr(config, "grayscale_obs", False),
        resize_obs=getattr(config, "resize_obs", False),
    )

    policy_model = _load_model(ckpt_path, config)
    policy_model.eval()

    # Load action labels from vec env wrapper if available; fallback to YAML
    action_labels: List[str] | None = None
    try:
        if hasattr(env, "get_action_labels"):
            labels = env.get_action_labels()  # type: ignore[attr-defined]
            if isinstance(labels, list):
                action_labels = [str(x) for x in labels]
    except Exception:
        action_labels = None
    if action_labels is None:
        env_info_yaml = _load_env_info_yaml(config.env_id)
        action_labels = _extract_action_labels(env_info_yaml)

    # Collect environment spec for summary tabs
    try:
        env_spec_obj = getattr(env, "get_spec", None)() if hasattr(env, "get_spec") else None
    except Exception:
        env_spec_obj = None
    try:
        reward_range = env.get_reward_range() if hasattr(env, "get_reward_range") else None
    except Exception:
        reward_range = None
    try:
        reward_threshold = env.get_reward_threshold() if hasattr(env, "get_reward_threshold") else None
    except Exception:
        reward_threshold = None
    try:
        input_dim = env.get_input_dim() if hasattr(env, "get_input_dim") else None
    except Exception:
        input_dim = None
    try:
        output_dim = env.get_output_dim() if hasattr(env, "get_output_dim") else None
    except Exception:
        output_dim = None
    try:
        observation_space_str = str(getattr(env, "observation_space", None))
    except Exception:
        observation_space_str = None
    try:
        action_space_str = str(getattr(env, "action_space", None))
    except Exception:
        action_space_str = None
    env_spec_summary: Dict[str, Any] = {
        "env_id": config.env_id,
        "n_envs": getattr(env, "num_envs", None),
        "observation_space": observation_space_str,
        "action_space": action_space_str,
        "reward_range": reward_range,
        "reward_threshold": reward_threshold,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "env_wrappers": getattr(config, "env_wrappers", None),
        "frame_stack": getattr(config, "frame_stack", None),
        "normalize_obs": getattr(config, "normalize_obs", None),
        "spec_id": getattr(getattr(env_spec_obj, "id", None), "__str__", lambda: None)() if env_spec_obj is not None else None,
        "action_labels": action_labels,
    }

    # Collect model spec for summary tabs
    try:
        num_params_total = int(sum(p.numel() for p in policy_model.parameters()))
        num_params_trainable = int(sum(p.numel() for p in policy_model.parameters() if p.requires_grad))
    except Exception:
        num_params_total = None
        num_params_trainable = None
    try:
        device = next(policy_model.parameters()).device.type  # type: ignore[attr-defined]
    except Exception:
        device = None
    model_spec_summary: Dict[str, Any] = {
        "algo_id": config.algo_id,
        "policy_class": type(policy_model).__name__,
        "device": device,
        "num_parameters_total": num_params_total,
        "num_parameters_trainable": num_params_trainable,
        "policy": getattr(config, "policy", None),
        "hidden_dims": getattr(config, "hidden_dims", None),
        "activation": getattr(config, "activation", None),
    }

    # Collect checkpoint metrics (from sidecar json or checkpoint contents)
    checkpoint_metrics_summary: Dict[str, Any] = {}
    try:
        import json
        ckpt_data = None
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt_data = None
        if isinstance(ckpt_data, dict):
            for k in [
                "epoch",
                "global_step",
                "total_timesteps",
                "best_eval_reward",
                "current_eval_reward",
                "is_best",
                "is_last",
                "is_threshold",
                "threshold_value",
            ]:
                if k in ckpt_data:
                    checkpoint_metrics_summary[k] = ckpt_data.get(k)
            # Merge metrics dict if present
            if isinstance(ckpt_data.get("metrics"), dict):
                checkpoint_metrics_summary["metrics"] = ckpt_data.get("metrics")
        # Sidecar JSON
        sidecar = ckpt_path.with_suffix(".json")
        if sidecar.exists():
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    sidecar_metrics = json.load(f)
                checkpoint_metrics_summary["sidecar_metrics"] = sidecar_metrics
            except Exception:
                pass
    except Exception:
        pass

    frames: List[np.ndarray] = []  # raw rendered frames for human monitoring
    processed_frames: List[np.ndarray] = []  # model-processed single-frame views (if applicable)
    stack_frames: List[np.ndarray] = []  # tiled frame-stack views (if applicable)

    # Heuristics and helpers to convert observations to displayable images
    def _to_uint8_img(arr: np.ndarray) -> np.ndarray:
        try:
            a = np.asarray(arr)
            if a.dtype != np.uint8:
                # Normalize floats
                a = a.astype(np.float32)
                maxv = float(np.nanmax(a)) if np.isfinite(a).any() else 0.0
                if maxv <= 1.0:
                    a = a * 255.0
                a = np.clip(a, 0, 255).astype(np.uint8)
            return a
        except Exception:
            return np.asarray(arr).astype(np.uint8, copy=False)

    def _obs_first_env(obs_any: Any) -> np.ndarray | None:
        try:
            arr = np.asarray(obs_any)
            if arr.ndim == 0:
                return None
            if arr.ndim >= 4:
                return arr[0]
            return arr
        except Exception:
            return None

    def _chw_to_hwc(img: np.ndarray) -> np.ndarray:
        # Accepts (C,H,W) and returns (H,W,C)
        if img.ndim == 3 and img.shape[0] <= 64 and img.shape[1] >= 8 and img.shape[2] >= 8:
            return np.transpose(img, (1, 2, 0))
        return img

    def _ensure_hwc(img: np.ndarray) -> np.ndarray:
        x = np.asarray(img)
        if x.ndim == 2:
            x = x[:, :, None]
        if x.ndim == 3:
            # Heuristic: if channels in first dim small, likely CHW
            if x.shape[0] <= 8 and (x.shape[1] >= 16 and x.shape[2] >= 16):
                x = _chw_to_hwc(x)
            # else assume HWC already
        return x

    def _to_rgb(img: np.ndarray) -> np.ndarray:
        x = _ensure_hwc(img)
        if x.ndim != 3:
            return None  # type: ignore[return-value]
        c = x.shape[2]
        if c == 1:
            x = np.repeat(x, 3, axis=2)
        elif c >= 3:
            x = x[:, :, :3]
        return _to_uint8_img(x)

    def _split_stack(img: np.ndarray, assume_rgb_groups: bool, n_stack_hint: int | None) -> List[np.ndarray]:
        """Split a stacked (C,H,W) or (H,W,C) image into a list of single-frame grayscale/RGB images.

        - For CHW with C > 3: if assume_rgb_groups and C % 3 == 0, split into groups of 3 channels; else split per channel as grayscale.
        - For HWC with C > 3: similar handling using last dimension.
        - For 2D grayscale with frame stacking that ended up as (H, W): cannot split; return [img].
        """
        try:
            x = np.asarray(img)
            if x.ndim == 2:
                return [x]
            # Normalize to HWC for consistent slicing
            if x.ndim == 3 and x.shape[0] <= 8 and x.shape[1] >= 16 and x.shape[2] >= 16:
                x = _chw_to_hwc(x)
            if x.ndim != 3:
                return []
            H, W, C = x.shape
            frames: List[np.ndarray] = []
            if assume_rgb_groups and C % 3 == 0:
                n = C // 3
                if n_stack_hint is not None and n_stack_hint > 0:
                    n = min(n, int(n_stack_hint))
                for i in range(n):
                    frames.append(x[:, :, 3 * i: 3 * (i + 1)])
            else:
                # Treat as n grayscale channels
                n = C
                if n_stack_hint is not None and n_stack_hint > 0:
                    n = min(n, int(n_stack_hint))
                for i in range(n):
                    frames.append(x[:, :, i:i + 1])
            return frames
        except Exception:
            return []

    def _make_grid(frames_hwc: List[np.ndarray], cols: int | None = None) -> np.ndarray | None:
        try:
            if not frames_hwc:
                return None
            imgs = [
                (_to_uint8_img(np.repeat(f, 3, axis=2)) if (f.ndim == 3 and f.shape[2] == 1) else _to_uint8_img(f))
                for f in frames_hwc
            ]
            H, W = imgs[0].shape[0], imgs[0].shape[1]
            # Ensure same size
            imgs = [img if (img.shape[0] == H and img.shape[1] == W) else np.resize(img, (H, W, 3)) for img in imgs]
            n = len(imgs)
            if cols is None:
                cols = n if n <= 4 else 4
            rows = int(np.ceil(n / float(cols)))
            grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
            for idx, img in enumerate(imgs):
                r = idx // cols
                c = idx % cols
                grid[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = img
            return grid
        except Exception:
            return None
    steps: List[Dict[str, Any]] = []

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    total_reward = 0.0
    t = 0

    # Buffers to compute MC returns and GAE after the episode
    rewards_buf: List[float] = []
    values_buf: List[float] = []
    dones_buf: List[bool] = []
    truncated_buf: List[bool] = []

    gamma: float = float(getattr(config, "gamma", 0.99))
    gae_lambda: float = float(getattr(config, "gae_lambda", 0.95))

    try:
        while t < max_steps:
            # Capture current frame BEFORE stepping (VecEnv resets on done)
            try:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    frames.append(frame)
            except Exception:
                pass

            # Build processed/stack views from current observation (pre-step)
            try:
                obs0 = _obs_first_env(obs)
                processed_img = None
                stack_img = None
                if isinstance(obs0, np.ndarray) and obs0.ndim in (2, 3):
                    hwc = _ensure_hwc(obs0)
                    processed_img = _to_rgb(hwc)
                    n_stack_hint = None
                    try:
                        n_stack_hint = int(getattr(config, "frame_stack", None) or 0)
                    except Exception:
                        n_stack_hint = None
                    split = _split_stack(hwc, assume_rgb_groups=True, n_stack_hint=n_stack_hint)
                    if isinstance(split, list) and len(split) >= 1:
                        grid = _make_grid(split, cols=None)
                        if grid is not None:
                            stack_img = grid
                if processed_img is not None:
                    processed_frames.append(processed_img)
                if stack_img is not None:
                    stack_frames.append(stack_img)
            except Exception:
                pass

            # Compute action from current obs
            with torch.no_grad():
                obs_t = _to_batch_obs(obs)
                dist, value = policy_model(obs_t)
                if deterministic:
                    action_t = getattr(dist, "mode", dist.mean)
                else:
                    action_t = dist.sample()
                action = action_t.squeeze().cpu().numpy()
                if np.ndim(action) == 0:
                    action = int(action.item())
                elif isinstance(action, np.ndarray) and action.shape == (1,):
                    action = action.item()
                action = _ensure_action_shape(env, action)

                probs = getattr(dist, "probs", None)
                if probs is not None:
                    probs = probs.squeeze(0).cpu().numpy().tolist()
                val = value.squeeze().item() if value is not None else None

            # Step environment
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated, truncated = done, False

            if isinstance(reward, (list, np.ndarray)):
                reward = float(reward[0])
            if isinstance(terminated, (list, np.ndarray)):
                terminated = bool(terminated[0])
            if isinstance(truncated, (list, np.ndarray)):
                truncated = bool(truncated[0])

            total_reward += float(reward)
            rewards_buf.append(float(reward))
            values_buf.append(float(val) if val is not None else 0.0)
            dones_buf.append(bool(terminated or truncated))
            truncated_buf.append(bool(truncated))

            steps.append({
                "step": t,
                "action": int(action[0]) if isinstance(action, np.ndarray) else int(action),
                "action_label": (action_labels[int(action[0]) if isinstance(action, np.ndarray) else int(action)] if action_labels is not None else None),
                "reward": float(reward),
                "cum_reward": float(total_reward),
                "value": float(val) if val is not None else None,
                "done": bool(terminated or truncated),
                "probs": probs,
            })

            t += 1

            if terminated or truncated:
                break
    finally:
        env.close()

    # Post-process per-step MC returns and GAE advantages
    if steps:
        T = len(rewards_buf)
        values_np = np.asarray(values_buf, dtype=np.float32)
        rewards_np = np.asarray(rewards_buf, dtype=np.float32)
        dones_np = np.asarray(dones_buf, dtype=bool)
        truncated_np = np.asarray(truncated_buf, dtype=bool)

        # MC returns (no bootstrap, pure discounted sum of observed rewards)
        mc_returns = compute_mc_returns(rewards_np, gamma)

        # Determine bootstrap value for the last step if episode truncated or hit max_steps
        last_next_value = 0.0
        ended_by_time = bool(truncated_np[-1]) or (not bool(dones_np[-1]))
        if ended_by_time:
            with torch.no_grad():
                last_v = policy_model(_to_batch_obs(obs))[1]
                if last_v is not None:
                    last_next_value = float(last_v.squeeze().item())

        gae_adv, _returns = compute_gae_advantages_and_returns(
            values=values_np,
            rewards=rewards_np,
            dones=dones_np,
            timeouts=truncated_np,
            last_value=last_next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Attach to steps
        for i in range(T):
            steps[i]["mc_return"] = float(mc_returns[i])
            steps[i]["gae_adv"] = float(gae_adv[i])

    # Only return processed/stack frames if lists are non-empty (else None)
    processed_out = processed_frames if processed_frames else None
    stack_out = stack_frames if stack_frames else None

    return frames, processed_out, stack_out, steps, {
        "total_reward": float(total_reward),
        "steps": t,
        "env_id": config.env_id,
        "algo_id": config.algo_id,
        "checkpoint_name": Path(ckpt_path).name,
        "env_spec": env_spec_summary,
        "model_spec": model_spec_summary,
        "checkpoint_metrics": checkpoint_metrics_summary,
    }


def build_ui(default_run_id: str = "latest-run"):
    import gradio as gr

    runs = list_runs()
    initial_run = default_run_id if default_run_id else (runs[0] if runs else "latest-run")
    labels, mapping, default_label = list_checkpoints_for_run(initial_run) if runs or initial_run == "latest-run" else ([], {}, None)

    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # Run Inspector
        Select a run and checkpoint to visualize an episode: frames, actions, rewards, values.
        """)

        with gr.Row():
            run_id = gr.Dropdown(
                label="Run ID",
                choices=["latest-run"] + runs,
                value=initial_run,
                interactive=True,
            )
            checkpoint = gr.Dropdown(
                label="Checkpoint",
                choices=labels,
                value=default_label,
                interactive=True,
            )
            deterministic = gr.Checkbox(label="Deterministic policy", value=False)
            max_steps = gr.Slider(label="Max steps", minimum=10, maximum=5000, value=1000, step=10)
            run_btn = gr.Button("Inspect")

        # Display the current frame with a per-step stats table on the right
        with gr.Row():
            with gr.Column(scale=7):
                display_mode = gr.Dropdown(
                    label="Display",
                    choices=["Rendered (raw)"],
                    value="Rendered (raw)",
                    interactive=True,
                )
                frame_image = gr.Image(label="Frame (raw)", height=400, type="numpy", image_mode="RGB")
            with gr.Column(scale=5):
                current_step_table = gr.Dataframe(
                    headers=["metric", "value"],
                    datatype=["str", "str"],
                    row_count=(0, "dynamic"),
                    col_count=(2, "fixed"),
                    label="",
                    show_label=False,
                    interactive=False,
                )

        # Horizontal navigation slider with integrated play/pause
        with gr.Row():
            with gr.Column(scale=9):
                frame_slider = gr.Slider(
                    label="Frame",
                    minimum=0,
                    maximum=0,
                    step=1,
                    value=0,
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=60):
                play_pause_btn = gr.Button(value="▶", variant="secondary")

        frames_state = gr.State([])  # active frames displayed
        frames_raw_state = gr.State([])  # type: ignore[var-annotated]
        frames_stack_state = gr.State([])  # type: ignore[var-annotated]
        has_stack_state = gr.State(False)
        index_state = gr.State(0)
        playing_state = gr.State(False)
        rows_state = gr.State([])  # type: ignore[var-annotated]
        steps_state = gr.State([])  # raw step dicts for precise right-side stats

        # Timer for autoplay (fallback if Timer doesn't exist in older Gradio)
        timer = None
        try:
            # Try to infer FPS for smoother playback
            inferred_fps = None
            try:
                # Quick attempt: load env_info YAML for the initial run's env
                cfg = _load_config_from_run(initial_run)
                if cfg and getattr(cfg, "env_id", None):
                    info = _load_env_info_yaml(cfg.env_id)
                    if isinstance(info, dict):
                        rfps = info.get("render_fps")
                        if isinstance(rfps, (int, float)) and rfps > 0:
                            inferred_fps = int(rfps)
            except Exception:
                inferred_fps = None

            TimerCls = getattr(gr, "Timer", None)
            if TimerCls is not None:
                # Default to 30 FPS if unknown
                fps_val = int(inferred_fps) if isinstance(inferred_fps, int) and inferred_fps > 0 else 30
                timer = TimerCls(1/float(fps_val))
        except Exception:
            timer = None
        # Table headers are reused for CSV export and for the current-step vertical view
        table_headers = [
            "done",
            "step",
            "action",
            "action_label",
            "probs",
            "reward",
            "cum_reward",
            "mc_return",
            "value",
            "gae_adv",
        ]

        with gr.Row():
            step_table = gr.Dataframe(
                headers=table_headers,
                datatype=[
                    "bool",   # done
                    "number", # step
                    "number", # action
                    "str",    # action_label
                    "str",    # probs (formatted string)
                    "number", # reward
                    "number", # cum_reward
                    "number", # mc_return
                    "number", # value
                    "number", # gae_adv
                ],
                row_count=(0, "dynamic"),
                col_count=(10, "fixed"),
                label="Per-step details",
                interactive=False,
            )
        with gr.Row():
            download_btn = gr.DownloadButton(label="Export CSV")
            csv_link = gr.File(label="CSV file", visible=False)
        with gr.Row():
            with gr.Tabs():
                with gr.Tab("Environment"):
                    env_spec_json = gr.JSON(label="Environment spec")
                with gr.Tab("Model"):
                    model_spec_json = gr.JSON(label="Model spec")
                with gr.Tab("Checkpoint"):
                    ckpt_metrics_json = gr.JSON(label="Checkpoint metrics")

        def _on_run_change(selected_run: str):
            try:
                labels, _, default_label = list_checkpoints_for_run(selected_run)
            except Exception:
                labels, default_label = [], None
            return gr.Dropdown(choices=labels, value=default_label)

        run_id.change(_on_run_change, inputs=run_id, outputs=checkpoint)

        def _format_stat_value(v: Any) -> str:
            try:
                if isinstance(v, float):
                    return f"{v:.3f}"
                if isinstance(v, (list, tuple)):
                    return "[" + ", ".join(_format_stat_value(x) for x in v) + "]"
                return str(v) if v is not None else ""
            except Exception:
                return str(v)

        def _verticalize_row(row: List[Any]) -> List[List[str]]:
            pairs: List[List[str]] = []
            for name, val in zip(table_headers, row):
                pairs.append([name, _format_stat_value(val)])
            return pairs

        def _inspect(rid: str, ckpt_label: str | None, det: bool, nsteps: int):
            frames_raw, _frames_proc_unused, frames_stack, steps, info = run_episode(rid, ckpt_label, det, int(nsteps))
            rows = []
            def _round3(x):
                try:
                    return round(float(x), 3)
                except Exception:
                    return x
            for s in steps:
                # Format probabilities, optionally with labels
                probs_fmt = None
                probs = s.get("probs")
                if isinstance(probs, list):
                    try:
                        probs_fmt = "[" + ", ".join(f"{float(p):.3f}" for p in probs) + "]"
                    except Exception:
                        probs_fmt = "[" + ", ".join(str(p) for p in probs) + "]"
                rows.append([
                    s["done"],
                    s["step"],
                    s["action"],
                    s.get("action_label"),
                    probs_fmt,
                    s["reward"],
                    s["cum_reward"],
                    _round3(s.get("mc_return", None)),
                    _round3(s.get("value", None)),
                    _round3(s.get("gae_adv", None)),
                ])
            # Initialize gallery selection, states, and play button label
            # Initialize the image (first frame) and slider range
            first_frame = frames_raw[0] if frames_raw else None
            # Compute vertical from raw dict for highest fidelity
            def _vertical_from_step_dict(step_dict: Dict[str, Any]) -> List[List[str]]:
                row_vals = [
                    bool(step_dict.get("done")),
                    int(step_dict.get("step", 0)),
                    step_dict.get("action"),
                    step_dict.get("action_label"),
                    step_dict.get("probs"),
                    step_dict.get("reward"),
                    step_dict.get("cum_reward"),
                    step_dict.get("mc_return"),
                    step_dict.get("value"),
                    step_dict.get("gae_adv"),
                ]
                return _verticalize_row(row_vals)
            # Compute available display modes
            has_stack = isinstance(frames_stack, list) and len(frames_stack) > 0
            choices = ["Rendered (raw)"] + (["Processed (stack)"] if has_stack else [])
            # Set active frames to raw by default
            return (
                gr.update(value="Rendered (raw)", choices=choices),  # display_mode
                gr.update(value=first_frame, label="Frame (raw)"),  # frame_image
                gr.update(minimum=0, maximum=(len(frames_raw) - 1 if frames_raw else 0), step=1, value=0),  # frame_slider
                (_vertical_from_step_dict(steps[0]) if steps else []), # current_step_table
                rows,                                       # step_table
                info.get("env_spec", {}),                   # env_spec_json
                info.get("model_spec", {}),                 # model_spec_json
                info.get("checkpoint_metrics", {}),         # ckpt_metrics_json
                frames_raw,                                 # frames_state (active = raw by default)
                0,                                          # index_state
                False,                                      # playing_state
                gr.update(value="▶"),                    # play_pause_btn label
                rows,                                       # rows_state
                steps,                                      # steps_state
                frames_raw,                                 # frames_raw_state
                (frames_stack or []),                       # frames_stack_state
                has_stack,                                  # has_stack_state
            )
        run_btn.click(
            _inspect,
            inputs=[run_id, checkpoint, deterministic, max_steps],
            outputs=[display_mode, frame_image, frame_slider, current_step_table, step_table, env_spec_json, model_spec_json, ckpt_metrics_json, frames_state, index_state, playing_state, play_pause_btn, rows_state, steps_state, frames_raw_state, frames_stack_state, has_stack_state],
        )

        # Keep rows_state in sync if the user edits the table
        def _on_table_change(df_rows):
            return df_rows
        step_table.change(_on_table_change, inputs=[step_table], outputs=[rows_state])

        # When a user selects a cell in the step table, select the corresponding frame in the gallery
        def _on_step_select(frames: List[np.ndarray], steps: List[Dict[str, Any]] | None, evt=None):
            """When a table cell is selected, select the corresponding frame in the gallery.

            Supports Gradio's SelectData event object or a dict payload with an
            "index" field. The index is generally a (row, col) pair.
            """
            row_idx = 0
            try:
                idx = None
                if isinstance(evt, dict):
                    idx = evt.get("index")
                else:
                    idx = getattr(evt, "index", None)

                if isinstance(idx, (list, tuple)) and len(idx) > 0:
                    row_idx = int(idx[0])
                elif isinstance(idx, int):
                    row_idx = int(idx)
                else:
                    row_idx = 0
            except Exception:
                row_idx = 0

            # Update the displayed image and slider; also pause playback and sync index state
            img = frames[row_idx] if (isinstance(frames, list) and 0 <= row_idx < len(frames)) else None
            if steps and 0 <= row_idx < len(steps):
                s = steps[row_idx]
                row_vals = [
                    bool(s.get("done")),
                    int(s.get("step", 0)),
                    s.get("action"),
                    s.get("action_label"),
                    s.get("probs"),
                    s.get("reward"),
                    s.get("cum_reward"),
                    s.get("mc_return"),
                    s.get("value"),
                    s.get("gae_adv"),
                ]
                row_val = _verticalize_row(row_vals)
            else:
                row_val = []
            return (
                gr.update(value=img),               # frame_image
                gr.update(value=row_idx),           # frame_slider
                gr.update(value=row_val),           # current_step_table
                row_idx,                            # index_state
                False,                              # playing_state
                gr.update(value="▶"),           # play_pause_btn label
            )
        step_table.select(_on_step_select, inputs=[frames_state, steps_state], outputs=[frame_image, frame_slider, current_step_table, index_state, playing_state, play_pause_btn])

        # Play/Pause handler
        def _on_play_pause(playing: bool):
            new_playing = not bool(playing)
            return new_playing, gr.update(value=("⏸" if new_playing else "▶"))

        def _on_slider_change(frames: List[np.ndarray], val: int, playing: bool, steps: List[Dict[str, Any]] | None):
            """Update current frame when user releases the slider.

            Preserve current playing state so programmatic slider updates during autoplay
            don't pause playback.
            """
            idx = int(val) if val is not None else 0
            img = frames[idx] if (isinstance(frames, list) and 0 <= idx < len(frames)) else None
            if steps and 0 <= idx < len(steps):
                s = steps[idx]
                row_vals = [
                    bool(s.get("done")),
                    int(s.get("step", 0)),
                    s.get("action"),
                    s.get("action_label"),
                    s.get("probs"),
                    s.get("reward"),
                    s.get("cum_reward"),
                    s.get("mc_return"),
                    s.get("value"),
                    s.get("gae_adv"),
                ]
                row_val = _verticalize_row(row_vals)
            else:
                row_val = []
            return gr.update(value=img), gr.update(value=row_val), idx, playing, gr.update(value=("⏸" if playing else "▶"))
        play_pause_btn.click(_on_play_pause, inputs=[playing_state], outputs=[playing_state, play_pause_btn])
        # While dragging, update the frame live for fast visual scanning (and pause playback)
        def _on_slider_input(frames: List[np.ndarray], val: int | float | None, steps: List[Dict[str, Any]] | None):
            # Use the slider's current value passed as an input to avoid any evt.value staleness
            idx = int(val) if val is not None else 0
            img = frames[idx] if (isinstance(frames, list) and 0 <= idx < len(frames)) else None
            # Pause while scrubbing for smoother UX and to avoid race with autoplay
            if steps and 0 <= idx < len(steps):
                s = steps[idx]
                row_vals = [
                    bool(s.get("done")),
                    int(s.get("step", 0)),
                    s.get("action"),
                    s.get("action_label"),
                    s.get("probs"),
                    s.get("reward"),
                    s.get("cum_reward"),
                    s.get("mc_return"),
                    s.get("value"),
                    s.get("gae_adv"),
                ]
                row_val = _verticalize_row(row_vals)
            else:
                row_val = []
            return gr.update(value=img), gr.update(value=row_val), idx, False, gr.update(value="▶")

        frame_slider.input(
            _on_slider_input,
            inputs=[frames_state, frame_slider, steps_state],
            outputs=[frame_image, current_step_table, index_state, playing_state, play_pause_btn],
        )

        # Use release instead of change to avoid triggering on programmatic updates from the timer
        frame_slider.release(_on_slider_change, inputs=[frames_state, frame_slider, playing_state, steps_state], outputs=[frame_image, current_step_table, index_state, playing_state, play_pause_btn])

        # Autoplay tick handler (only if timer available)
        def _on_tick(frames: List[np.ndarray], idx: int, playing: bool, steps: List[Dict[str, Any]] | None):
            if not playing or not frames:
                return gr.update(), gr.update(), gr.update(), idx, playing, gr.update()
            if int(idx) < len(frames) - 1:
                new_idx = int(idx) + 1
                if steps and 0 <= new_idx < len(steps):
                    s = steps[new_idx]
                    row_vals = [
                        bool(s.get("done")),
                        int(s.get("step", 0)),
                        s.get("action"),
                        s.get("action_label"),
                        s.get("probs"),
                        s.get("reward"),
                        s.get("cum_reward"),
                        s.get("mc_return"),
                        s.get("value"),
                        s.get("gae_adv"),
                    ]
                    row_val = _verticalize_row(row_vals)
                else:
                    row_val = []
                return gr.update(value=frames[new_idx]), gr.update(value=new_idx), gr.update(value=row_val), new_idx, True, gr.update(value="⏸")
            # Reached end: stop
            last_idx = len(frames) - 1
            if steps and 0 <= last_idx < len(steps):
                s = steps[last_idx]
                row_vals = [
                    bool(s.get("done")),
                    int(s.get("step", 0)),
                    s.get("action"),
                    s.get("action_label"),
                    s.get("probs"),
                    s.get("reward"),
                    s.get("cum_reward"),
                    s.get("mc_return"),
                    s.get("value"),
                    s.get("gae_adv"),
                ]
                row_val = _verticalize_row(row_vals)
            else:
                row_val = []
            return gr.update(value=frames[last_idx]), gr.update(value=last_idx), gr.update(value=row_val), last_idx, False, gr.update(value="▶")

        if timer is not None:
            timer.tick(_on_tick, inputs=[frames_state, index_state, playing_state, steps_state], outputs=[frame_image, frame_slider, current_step_table, index_state, playing_state, play_pause_btn])

        # Display mode switcher
        def _on_display_mode(mode: str, raw: List[np.ndarray], stack: List[np.ndarray], steps: List[Dict[str, Any]] | None):
            mode = str(mode or "Rendered (raw)")
            if mode == "Processed (stack)" and isinstance(stack, list) and len(stack) > 0:
                active = stack
                label = "Frame (processed stack)"
            else:
                active = raw if isinstance(raw, list) else []
                label = "Frame (raw)"
            first = active[0] if active else None
            # Build current step stats for index 0
            if steps and len(steps) > 0:
                s0 = steps[0]
                row_vals = [
                    bool(s0.get("done")),
                    int(s0.get("step", 0)),
                    s0.get("action"),
                    s0.get("action_label"),
                    s0.get("probs"),
                    s0.get("reward"),
                    s0.get("cum_reward"),
                    s0.get("mc_return"),
                    s0.get("value"),
                    s0.get("gae_adv"),
                ]
                vert = _verticalize_row(row_vals)
            else:
                vert = []
            return (
                gr.update(value=first, label=label),
                gr.update(minimum=0, maximum=(len(active) - 1 if active else 0), step=1, value=0),
                active,
                0,
                False,
                gr.update(value="▶"),
                gr.update(value=vert),
            )

        display_mode.change(
            _on_display_mode,
            inputs=[display_mode, frames_raw_state, frames_stack_state, steps_state],
            outputs=[frame_image, frame_slider, frames_state, index_state, playing_state, play_pause_btn, current_step_table],
        )

        # CSV export handler
        def _export_csv(rows: List[List[Any]] | None, rid: str, ckpt_label: str | None):
            import csv
            import tempfile
            import os

            # Normalize possible pandas.DataFrame or None → list of lists
            if rows is None:
                return None, gr.update(visible=False)
            try:
                # Detect DataFrame by attribute presence to avoid strict import dependency
                if hasattr(rows, "empty") and hasattr(rows, "to_numpy"):
                    try:
                        import pandas as pd  # type: ignore
                        rows = rows.where(pd.notnull(rows), None).to_numpy().tolist()  # type: ignore[attr-defined]
                    except Exception:
                        rows = rows.to_numpy().tolist()  # type: ignore[assignment]
            except Exception:
                pass

            # Ensure rows is a list and check emptiness safely
            if isinstance(rows, list):
                if len(rows) == 0:
                    return None, gr.update(visible=False)
            else:
                try:
                    rows = list(rows)  # type: ignore[arg-type]
                except Exception:
                    return None, gr.update(visible=False)
                if len(rows) == 0:
                    return None, gr.update(visible=False)

            safe_rid = str(rid).replace("/", "-").replace(" ", "_")
            safe_ckpt = str(ckpt_label or "ckpt").replace("/", "-").replace(" ", "_")
            file_name = f"{safe_rid}_{safe_ckpt}_steps.csv"
            tmp_dir = tempfile.gettempdir()
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(table_headers)
                writer.writerows(rows)
            # Return for both: triggers download and shows a visible link as fallback
            return file_path, gr.update(value=file_path, visible=True)
        # Wire the download to the same button so clicking it both generates and downloads the CSV,
        # and also shows a link in case auto-download is blocked by the browser.
        download_btn.click(_export_csv, inputs=[rows_state, run_id, checkpoint], outputs=[download_btn, csv_link])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio app to inspect a run's checkpoints visually.")
    parser.add_argument("--run-id", type=str, default="latest-run", help="Run ID under runs/ (default: latest-run)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    demo = build_ui(args.run_id)
    demo.launch(server_port=args.port, server_name=args.host, share=args.share)


if __name__ == "__main__":
    main()
