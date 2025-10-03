#!/usr/bin/env python3
"""Inspector application logic (run via: python inspector.py [args])."""

from __future__ import annotations

import argparse

# Avoid macOS AppKit main-thread violations when environments initialize
# Pygame/SDL from Gradio worker threads. Using the headless SDL drivers
# prevents Cocoa window/menu initialization on non-main threads.
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

if platform.system() == "Darwin":
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# UI constants to keep strings consistent
DISPLAY_RAW = "Rendered (raw)"
DISPLAY_STACK = "Processed (stack)"
FRAME_LABEL_RAW = "Frame (raw)"
FRAME_LABEL_STACK = "Frame (processed stack)"
PLAY_ICON = "\u25B6"  # ▶
PAUSE_ICON = "\u23F8"  # ⏸

from utils.policy_factory import load_policy_model_from_checkpoint
from utils.rollouts import (
    compute_batched_gae_advantages_and_returns,
    compute_batched_mc_returns,
)
from utils.run import Run, list_run_ids


def _to_batched_array(arr: np.ndarray, dtype) -> np.ndarray:
    """Convert 1D array to batched (T, 1) format."""
    return np.asarray(arr, dtype=dtype).reshape(-1, 1)


def compute_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted Monte Carlo returns for a single trajectory (batch size 1 wrapper)."""
    rewards_b = _to_batched_array(rewards, np.float32)
    T = rewards_b.shape[0]
    dones_b = np.zeros((T, 1), dtype=bool)
    timeouts_b = np.zeros((T, 1), dtype=bool)
    returns_b = compute_batched_mc_returns(rewards_b, dones_b, timeouts_b, float(gamma))
    return returns_b.reshape(-1)


def compute_gae_advantages_and_returns(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timeouts: np.ndarray | None,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE(λ) advantages and returns for a single (T,) trajectory (batch size 1 wrapper)."""
    values_b = _to_batched_array(values, np.float32)
    rewards_b = _to_batched_array(rewards, np.float32)
    dones_b = _to_batched_array(dones, bool)
    timeouts_b = np.zeros_like(dones_b, dtype=bool) if timeouts is None else _to_batched_array(timeouts, bool)
    last_values = np.asarray([float(last_value)], dtype=np.float32)

    adv_b, ret_b = compute_batched_gae_advantages_and_returns(
        values=values_b,
        rewards=rewards_b,
        dones=dones_b,
        timeouts=timeouts_b,
        last_values=last_values,
        bootstrapped_next_values=None,
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
    )
    return adv_b.reshape(-1), ret_b.reshape(-1)
# Note: Action labels/spec/fps fallbacks are provided by VecInfoWrapper methods.
# Avoid duplicating YAML parsing here; call env.get_action_labels(), env.get_spec(), etc.


def _to_batch_obs(obs) -> torch.Tensor:
    if isinstance(obs, np.ndarray) and obs.ndim > 1:
        return torch.as_tensor(obs)
    return torch.as_tensor(obs).unsqueeze(0)


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
    run = Run.load(run_id)
    config = run.load_config()

    #labels, _, default_label = run.list_checkpoints(), {}, "@best"
    checkpoint_dir = run.checkpoints_dir / checkpoint_label

    from utils.environment import build_env_from_config

    env = build_env_from_config(
        config,
        n_envs=1,
        render_mode="rgb_array",
        vectorization_mode='sync',
    )

    # Get checkpoint path with backward compatibility for old policy.ckpt format
    checkpoint_path = run._get_checkpoint_path(checkpoint_dir)
    policy_model, ckpt_data = load_policy_model_from_checkpoint(checkpoint_path, env, config)

    # Load action labels from vec env wrapper if available
    act_labels_raw = env.get_action_labels()
    action_labels: List[str] | None = None
    if act_labels_raw is not None:
        if isinstance(act_labels_raw, dict):
            # Convert dict {0: 'NOOP', 1: 'RIGHT', ...} to list ['NOOP', 'RIGHT', ...]
            max_idx = max(act_labels_raw.keys()) if act_labels_raw else -1
            action_labels = [str(act_labels_raw.get(i, i)) for i in range(max_idx + 1)]
        else:
            action_labels = [str(x) for x in act_labels_raw]

    # Collect environment spec for summary tabs (VecInfoWrapper exposes safe helpers)
    env_spec_obj = env.get_spec()
    reward_range = env.get_reward_range()
    reward_threshold = env.get_return_threshold()
    observation_space_str = str(getattr(env, "observation_space", None))
    action_space_str = str(getattr(env, "action_space", None))
    env_spec_summary: Dict[str, Any] = {
        "env_id": config.env_id,
        "n_envs": getattr(env, "num_envs", None),
        "observation_space": observation_space_str,
        "action_space": action_space_str,
        "reward_range": reward_range,
        "reward_threshold": reward_threshold,
        "env_wrappers": getattr(config, "env_wrappers", None),
        "frame_stack": getattr(config, "frame_stack", None),
        "normalize_obs": getattr(config, "normalize_obs", None),
        "normalize_reward": getattr(config, "normalize_reward", None),
        "spec_id": env_spec_obj.id,
        "action_labels": action_labels,
    }

    # Collect model spec for summary tabs
    num_params_total = int(sum(p.numel() for p in policy_model.parameters()))
    num_params_trainable = int(sum(p.numel() for p in policy_model.parameters() if p.requires_grad))
    device = next(policy_model.parameters()).device.type
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
    import json
    raw_ckpt = ckpt_data
    if isinstance(raw_ckpt, dict):
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
            if k in raw_ckpt:
                checkpoint_metrics_summary[k] = raw_ckpt.get(k)
        # Merge metrics dict if present
        metrics_dict = raw_ckpt.get("metrics") if isinstance(raw_ckpt, dict) else None
        if isinstance(metrics_dict, dict):
            checkpoint_metrics_summary["metrics"] = metrics_dict
    # Sidecar JSON
    sidecar = checkpoint_path / "metrics.json"
    if sidecar.exists():
        with open(sidecar, "r", encoding="utf-8") as f:
            sidecar_metrics = json.load(f)
        checkpoint_metrics_summary["sidecar_metrics"] = sidecar_metrics

    frames: List[np.ndarray] = []  # raw rendered frames for human monitoring
    stack_frames: List[np.ndarray] = []  # tiled frame-stack views (if applicable)

    # Heuristics and helpers to convert observations to displayable images
    def _to_uint8_img(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            # Normalize floats
            a = a.astype(np.float32)
            maxv = float(np.nanmax(a)) if np.isfinite(a).any() else 0.0
            if maxv <= 1.0:
                a = a * 255.0
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    def _obs_first_env(obs_any: Any) -> np.ndarray | None:
        arr = np.asarray(obs_any)
        if arr.ndim == 0:
            return None
        if arr.ndim >= 4:
            return arr[0]
        return arr

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

    # No single-frame processed RGB conversion needed; UI focuses on raw render and stack grid.

    def _split_stack(img: np.ndarray, assume_rgb_groups: bool, n_stack_hint: int | None) -> List[np.ndarray]:
        """Split a stacked (C,H,W) or (H,W,C) image into a list of single-frame grayscale/RGB images.

        - For CHW with C > 3: if assume_rgb_groups and C % 3 == 0, split into groups of 3 channels; else split per channel as grayscale.
        - For HWC with C > 3: similar handling using last dimension.
        - For 2D grayscale with frame stacking that ended up as (H, W): cannot split; return [img].
        """
        x = np.asarray(img)
        if x.ndim == 2:
            return [x]
        # Normalize to HWC for consistent slicing
        if x.ndim == 3 and x.shape[0] <= 8 and x.shape[1] >= 16 and x.shape[2] >= 16:
            x = _chw_to_hwc(x)
        if x.ndim != 3:
            return []
        H, W, C = x.shape
        # Determine split size and stride
        use_rgb = assume_rgb_groups and C % 3 == 0
        stride = 3 if use_rgb else 1
        n = (C // stride)
        if n_stack_hint is not None and n_stack_hint > 0:
            n = min(n, int(n_stack_hint))
        return [x[:, :, i * stride: (i + 1) * stride] for i in range(n)]

    def _make_grid(frames_hwc: List[np.ndarray], cols: int | None = None) -> np.ndarray | None:
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

    def _vector_stack_to_frames(vec: np.ndarray, n_stack_hint: int | None, row_height: int = 8) -> List[np.ndarray]:
        """Represent a stacked 1D observation as a list of grayscale bar images.

        Splits the vector into ``n_stack_hint`` equal chunks (when possible) and
        turns each chunk into an ``(row_height, chunk_len, 1)`` image with values
        normalized per-chunk to [0, 1] for visibility. Falls back to a single
        bar if splitting is not possible or the hint is invalid.
        """
        v = np.asarray(vec).reshape(-1)
        n = int(n_stack_hint) if (n_stack_hint is not None and int(n_stack_hint) > 1) else 1
        L = v.shape[0]
        # Split into chunks
        seg = L // n
        chunks = [v[i * seg:(i + 1) * seg] for i in range(n)] if n > 1 and L % n == 0 else [v]
        # Convert each chunk to a normalized bar image
        frames: List[np.ndarray] = []
        for chunk in chunks:
            c = np.asarray(chunk, dtype=np.float32)
            c_min = float(np.nanmin(c)) if c.size > 0 else 0.0
            c_max = float(np.nanmax(c)) if c.size > 0 else 1.0
            # Normalize to [0,1] per-chunk; avoid div by zero
            denom = (c_max - c_min) if (c_max - c_min) > 1e-9 else 1.0
            c01 = (c - c_min) / denom
            row = c01.reshape(1, -1, 1)  # (1, W, 1)
            img = np.repeat(row, row_height, axis=0)  # (H, W, 1)
            frames.append(img)
        return frames
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
    # Used to decide how to visualize non-image observations
    obs_type_cfg = str(getattr(config, "obs_type", "")).lower()
    # Derive a fixed stacking hint once (config frame_stack is now always coherent)
    n_stack_hint = int(getattr(config, "frame_stack", None) or 0)
    has_stack_hint = bool(n_stack_hint is not None and int(n_stack_hint) > 1)

    try:
        while t < max_steps:
            # Capture current frame BEFORE stepping (VecEnv resets on done)
            frame = env.render()
            # Handle vectorized env render output (tuple/list of frames)
            if isinstance(frame, (tuple, list)) and len(frame) > 0:
                frame = frame[0]
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)

            # Build processed/stack views from current observation (pre-step)
            obs0 = _obs_first_env(obs)
            stack_img = None
            if has_stack_hint:
                # Build a stack grid from the observation channels when frame stacking is enabled
                if isinstance(obs0, np.ndarray) and obs0.ndim in (2, 3):
                    hwc = _ensure_hwc(obs0)
                    split = _split_stack(hwc, assume_rgb_groups=True, n_stack_hint=n_stack_hint)
                    if isinstance(split, list) and len(split) >= 2:
                        stack_img = _make_grid(split, cols=None)
                # Vector observation fallback
                elif isinstance(obs0, np.ndarray) and (obs0.ndim == 1 or (obs0.ndim == 2 and 1 in obs0.shape)) and obs_type_cfg not in {"ram", "objects"}:
                    vec = obs0.reshape(-1)
                    frames_vec = _vector_stack_to_frames(vec, n_stack_hint=n_stack_hint, row_height=8)
                    if frames_vec:
                        stack_img = _make_grid(frames_vec, cols=None)
                # As a final fallback for non-image observations, build a grid from the last N rendered frames
                elif isinstance(frames, list) and len(frames) >= 1:
                    last = frames[-int(n_stack_hint):]
                    stack_img = _make_grid([_ensure_hwc(f) for f in last], cols=None)

            if stack_img is not None:
                stack_frames.append(stack_img)

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

            # Unwrap vectorized outputs
            def _unwrap_vec(val):
                return val[0] if isinstance(val, (list, np.ndarray)) else val

            reward = float(_unwrap_vec(reward))
            terminated = bool(_unwrap_vec(terminated))
            truncated = bool(_unwrap_vec(truncated))
            done = terminated or truncated

            total_reward += reward
            rewards_buf.append(reward)
            values_buf.append(float(val) if val is not None else 0.0)
            dones_buf.append(done)
            truncated_buf.append(truncated)

            action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            steps.append({
                "step": t,
                "action": action_idx,
                "action_label": (action_labels[action_idx] if action_labels is not None and 0 <= action_idx < len(action_labels) else None),
                "reward": reward,
                "cum_reward": total_reward,
                "value": float(val) if val is not None else None,
                "done": done,
                "probs": probs,
            })

            t += 1

            if done:
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

    # Only return stack frames if lists are non-empty (else None)
    stack_out = stack_frames if stack_frames else None

    return frames, None, stack_out, steps, {
        "total_reward": float(total_reward),
        "steps": t,
        "env_id": config.env_id,
        "algo_id": config.algo_id,
        "checkpoint_name": Path(checkpoint_path).name,
        "env_spec": env_spec_summary,
        "model_spec": model_spec_summary,
        "checkpoint_metrics": checkpoint_metrics_summary,
    }


def build_ui(default_run_id: str = "@last"):
    import gradio as gr

    runs = list_run_ids()
    initial_run = default_run_id if default_run_id else (runs[0] if runs else "@last")

    def _checkpoint_choices_for_run(run_identifier: str):
        checkpoints = Run.load(run_identifier).list_checkpoints()
        return checkpoints, {}, "@best"

    labels, _, default_label = _checkpoint_choices_for_run(initial_run)

    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # Run Inspector
        Select a run and checkpoint to visualize an episode: frames, actions, rewards, values.
        """)

        with gr.Row():
            run_id = gr.Dropdown(
                label="Run ID",
                choices=["@last"] + runs,
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
                frame_image = gr.Image(label=FRAME_LABEL_RAW, height=400, type="numpy", image_mode="RGB")
                display_mode = gr.Radio(
                    choices=[DISPLAY_RAW],
                    value=DISPLAY_RAW,
                    interactive=True,
                    type="value",
                    show_label=False,
                    container=False,
                )
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
                play_pause_btn = gr.Button(value=PLAY_ICON, variant="secondary")

        frames_state = gr.State([])  # active frames displayed
        frames_raw_state = gr.State([])  # type: ignore[var-annotated]
        frames_stack_state = gr.State([])  # type: ignore[var-annotated]
        index_state = gr.State(0)
        playing_state = gr.State(False)
        rows_state = gr.State([])  # type: ignore[var-annotated]
        steps_state = gr.State([])  # raw step dicts for precise right-side stats

        # Timer for autoplay (fallback if Timer doesn't exist in older Gradio)
        timer = None
        TimerCls = getattr(gr, "Timer", None)
        if TimerCls is not None:
            timer = TimerCls(1/30.0)  # default to 30 FPS
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
            labels, _, default_label = _checkpoint_choices_for_run(selected_run)
            return gr.Dropdown(choices=labels, value=default_label)

        run_id.change(_on_run_change, inputs=run_id, outputs=checkpoint)

        def _format_stat_value(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.3f}"
            if isinstance(v, (list, tuple)):
                return "[" + ", ".join(_format_stat_value(x) for x in v) + "]"
            return str(v) if v is not None else ""

        def _round3(x):
            return round(float(x), 3)

        def _format_probs(probs: Any) -> str | None:
            if isinstance(probs, list):
                return "[" + ", ".join(f"{float(p):.3f}" for p in probs) + "]"
            return str(probs) if probs is not None else None

        def _verticalize_row(row: List[Any]) -> List[List[str]]:
            pairs: List[List[str]] = []
            for name, val in zip(table_headers, row):
                pairs.append([name, _format_stat_value(val)])
            return pairs

        # DRY helpers: build consistent row values and vertical pairs from a step dict
        def _row_vals_from_step_dict(step_dict: Dict[str, Any] | None) -> List[Any]:
            if not isinstance(step_dict, dict):
                return []
            return [
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

        def _vertical_from_steps_index(steps: List[Dict[str, Any]] | None, idx: int) -> List[List[str]]:
            if steps and 0 <= int(idx) < len(steps):
                return _verticalize_row(_row_vals_from_step_dict(steps[int(idx)]))
            return []

        def _initial_display_state(active_frames: List[np.ndarray] | None, frame_label: str, steps: List[Dict[str, Any]] | None):
            first = (active_frames[0] if active_frames else None) if isinstance(active_frames, list) else None
            slider = gr.update(minimum=0, maximum=(len(active_frames) - 1 if active_frames else 0), step=1, value=0)
            vert = _vertical_from_steps_index(steps, 0)
            return (
                gr.update(value=first, label=frame_label),
                slider,
                (active_frames or []),
                0,
                False,
                gr.update(value=PLAY_ICON),
                gr.update(value=vert),
            )

        def _table_row_from_step(s: Dict[str, Any]) -> List[Any]:
            return [
                s["done"],
                s["step"],
                s["action"],
                s.get("action_label"),
                _format_probs(s.get("probs")),
                s["reward"],
                s["cum_reward"],
                _round3(s.get("mc_return", None)),
                _round3(s.get("value", None)),
                _round3(s.get("gae_adv", None)),
            ]

        def _inspect(rid: str, ckpt_label: str | None, det: bool, nsteps: int):
            frames_raw, _frames_proc_unused, frames_stack, steps, info = run_episode(rid, ckpt_label, det, int(nsteps))
            rows = [_table_row_from_step(s) for s in steps]
            # Initialize gallery selection, states, play button, and slider range
            # Compute available display modes
            has_stack = isinstance(frames_stack, list) and len(frames_stack) > 0
            choices = [DISPLAY_RAW] + ([DISPLAY_STACK] if has_stack else [])
            display_value, frame_label, active_frames = (DISPLAY_STACK, FRAME_LABEL_STACK, frames_stack) if has_stack else (DISPLAY_RAW, FRAME_LABEL_RAW, frames_raw)

            frame_img_upd, slider_upd, active_list, idx_val, playing_val, play_btn_upd, vert_pairs = _initial_display_state(active_frames, frame_label, steps)

            return (
                gr.update(value=display_value, choices=choices),
                frame_img_upd,
                slider_upd,
                vert_pairs,
                rows,
                info.get("env_spec", {}),
                info.get("model_spec", {}),
                info.get("checkpoint_metrics", {}),
                active_list,
                idx_val,
                playing_val,
                play_btn_upd,
                rows,
                steps,
                frames_raw,
                (frames_stack or []),
            )
        run_btn.click(
            _inspect,
            inputs=[run_id, checkpoint, deterministic, max_steps],
            outputs=[display_mode, frame_image, frame_slider, current_step_table, step_table, env_spec_json, model_spec_json, ckpt_metrics_json, frames_state, index_state, playing_state, play_pause_btn, rows_state, steps_state, frames_raw_state, frames_stack_state],
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

            # Update the displayed image and slider; also pause playback and sync index state
            img = frames[row_idx] if (isinstance(frames, list) and 0 <= row_idx < len(frames)) else None
            row_val = _vertical_from_steps_index(steps, row_idx)
            return (
                gr.update(value=img),               # frame_image
                gr.update(value=row_idx),           # frame_slider
                gr.update(value=row_val),           # current_step_table
                row_idx,                            # index_state
                False,                              # playing_state
                gr.update(value=PLAY_ICON),         # play_pause_btn label
            )
        step_table.select(_on_step_select, inputs=[frames_state, steps_state], outputs=[frame_image, frame_slider, current_step_table, index_state, playing_state, play_pause_btn])

        # Play/Pause handler
        def _on_play_pause(playing: bool):
            new_playing = not bool(playing)
            return new_playing, gr.update(value=(PAUSE_ICON if new_playing else PLAY_ICON))

        # Core helper to build current-frame UI updates from frames/index/steps
        def _current_view_updates(frames: List[np.ndarray], idx: int | float | None, steps: List[Dict[str, Any]] | None):
            i = int(idx) if idx is not None else 0
            img = frames[i] if (isinstance(frames, list) and 0 <= i < len(frames)) else None
            row_val = _vertical_from_steps_index(steps, i)
            return img, row_val, i

        def _build_slider_output(img, row_val, i, playing: bool):
            """Build standard tuple for slider change handlers."""
            return gr.update(value=img), gr.update(value=row_val), i, playing, gr.update(value=(PAUSE_ICON if playing else PLAY_ICON))

        def _on_slider_change(frames: List[np.ndarray], val: int, playing: bool, steps: List[Dict[str, Any]] | None):
            """Update current frame when user releases the slider, preserving play state."""
            img, row_val, i = _current_view_updates(frames, val, steps)
            return _build_slider_output(img, row_val, i, playing)
        play_pause_btn.click(_on_play_pause, inputs=[playing_state], outputs=[playing_state, play_pause_btn])
        # While dragging, update the frame live for fast visual scanning (and pause playback)
        def _on_slider_input(frames: List[np.ndarray], val: int | float | None, steps: List[Dict[str, Any]] | None):
            # Pause while scrubbing for smoother UX and to avoid race with autoplay
            img, row_val, i = _current_view_updates(frames, val, steps)
            return _build_slider_output(img, row_val, i, False)

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
            new_idx = int(idx) + 1 if int(idx) < len(frames) - 1 else len(frames) - 1
            img, row_val, _ = _current_view_updates(frames, new_idx, steps)
            still_playing = int(idx) < len(frames) - 1
            return gr.update(value=img), gr.update(value=new_idx), gr.update(value=row_val), new_idx, still_playing, gr.update(value=(PAUSE_ICON if still_playing else PLAY_ICON))

        if timer is not None:
            timer.tick(_on_tick, inputs=[frames_state, index_state, playing_state, steps_state], outputs=[frame_image, frame_slider, current_step_table, index_state, playing_state, play_pause_btn])

        # Display mode switcher
        def _on_display_mode(mode: str, raw: List[np.ndarray], stack: List[np.ndarray], steps: List[Dict[str, Any]] | None):
            mode = str(mode or DISPLAY_RAW)
            use_stack = mode == DISPLAY_STACK and isinstance(stack, list) and len(stack) > 0
            active = stack if use_stack else (raw if isinstance(raw, list) else [])
            label = FRAME_LABEL_STACK if use_stack else FRAME_LABEL_RAW
            return _initial_display_state(active, label, steps)

        display_mode.change(
            _on_display_mode,
            inputs=[display_mode, frames_raw_state, frames_stack_state, steps_state],
            outputs=[frame_image, frame_slider, frames_state, index_state, playing_state, play_pause_btn, current_step_table],
        )

        def _sanitize_filename(s: str) -> str:
            """Sanitize string for use in filenames."""
            return str(s).replace("/", "-").replace(" ", "_")

        # CSV export handler
        def _export_csv(rows: List[List[Any]] | None, rid: str, ckpt_label: str | None):
            import csv
            import tempfile

            # Normalize possible pandas.DataFrame or None → list of lists
            if rows is None:
                return None, gr.update(visible=False)
            # Detect DataFrame by attribute presence to avoid strict import dependency
            if hasattr(rows, "empty") and hasattr(rows, "to_numpy"):
                try:
                    import pandas as pd  # type: ignore
                    rows = rows.where(pd.notnull(rows), None).to_numpy().tolist()
                except ImportError:
                    rows = rows.to_numpy().tolist()  # type: ignore[assignment]

            # Ensure rows is a list and check emptiness safely
            if not isinstance(rows, list):
                rows = list(rows)  # type: ignore[arg-type]
            if len(rows) == 0:
                return None, gr.update(visible=False)

            safe_rid = _sanitize_filename(rid)
            safe_ckpt = _sanitize_filename(ckpt_label or "ckpt")
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
    parser.add_argument("--run-id", type=str, default="@last", help="Run ID under runs/ (default: @last)") # TODO: remove @last harcode (this is a Run SOC)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    demo = build_ui(args.run_id)
    demo.launch(server_port=args.port, server_name=args.host, share=args.share)


if __name__ == "__main__":
    main()
