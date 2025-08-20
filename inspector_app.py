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
    deterministic: bool = True,
    max_steps: int = 1000,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
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
    )

    policy_model = _load_model(ckpt_path, config)
    policy_model.eval()

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

    frames: List[np.ndarray] = []
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
            # Record for MC/GAE
            rewards_buf.append(float(reward))
            values_buf.append(float(val) if val is not None else 0.0)
            dones_buf.append(bool(terminated or truncated))
            truncated_buf.append(bool(truncated))

            frame = env.render()
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)

            steps.append({
                "step": t,
                "action": int(action[0]) if isinstance(action, np.ndarray) else int(action),
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

    return frames, steps, {
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
            deterministic = gr.Checkbox(label="Deterministic policy", value=True)
            max_steps = gr.Slider(label="Max steps", minimum=10, maximum=5000, value=1000, step=10)
            run_btn = gr.Button("Inspect")

        # Display only the current frame (hide the thumbnail strip previously provided by Gallery)
        with gr.Row():
            frame_image = gr.Image(label="Frame", height=400, type="numpy", image_mode="RGB")

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

        frames_state = gr.State([])  # type: ignore[var-annotated]
        index_state = gr.State(0)
        playing_state = gr.State(False)
        rows_state = gr.State([])  # type: ignore[var-annotated]

        # Timer for autoplay (fallback if Timer doesn't exist in older Gradio)
        timer = None
        try:
            TimerCls = getattr(gr, "Timer", None)
            if TimerCls is not None:
                # ~30 FPS playback
                timer = TimerCls(1/30)
        except Exception:
            timer = None
        # Table headers are reused for CSV export
        table_headers = [
            "done",
            "step",
            "probs",
            "action",
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
                    "str",    # probs (formatted string)
                    "number", # reward
                    "number", # cum_reward
                    "number", # mc_return
                    "number", # value
                    "number", # gae_adv
                ],
                row_count=(0, "dynamic"),
                col_count=(9, "fixed"),
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

        def _inspect(rid: str, ckpt_label: str | None, det: bool, nsteps: int):
            frames, steps, info = run_episode(rid, ckpt_label, det, int(nsteps))
            rows = []
            for s in steps:
                rows.append([
                    s["done"],
                    s["step"],
                    "[" + ", ".join(f"{p:.3f}" for p in (s["probs"] or [])) + "]" if s["probs"] else None,
                    s["action"],
                    s["reward"],
                    s["cum_reward"],
                    s.get("mc_return", None),
                    s.get("value", None),
                    s.get("gae_adv", None),
                ])
            # Initialize gallery selection, states, and play button label
            # Initialize the image (first frame) and slider range
            first_frame = frames[0] if frames else None
            return (
                gr.update(value=first_frame),  # frame_image
                gr.update(minimum=0, maximum=(len(frames) - 1 if frames else 0), step=1, value=0),  # frame_slider
                rows,                                       # step_table
                info.get("env_spec", {}),                   # env_spec_json
                info.get("model_spec", {}),                 # model_spec_json
                info.get("checkpoint_metrics", {}),         # ckpt_metrics_json
                frames,                                     # frames_state
                0,                                          # index_state
                False,                                      # playing_state
                gr.update(value="▶"),                    # play_pause_btn label
                rows,                                       # rows_state
            )
        run_btn.click(
            _inspect,
            inputs=[run_id, checkpoint, deterministic, max_steps],
            outputs=[frame_image, frame_slider, step_table, env_spec_json, model_spec_json, ckpt_metrics_json, frames_state, index_state, playing_state, play_pause_btn, rows_state],
        )

        # Keep rows_state in sync if the user edits the table
        def _on_table_change(df_rows):
            return df_rows
        step_table.change(_on_table_change, inputs=[step_table], outputs=[rows_state])

        # When a user selects a cell in the step table, select the corresponding frame in the gallery
        def _on_step_select(frames: List[np.ndarray], evt=None):
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
            return (
                gr.update(value=img),               # frame_image
                gr.update(value=row_idx),           # frame_slider
                row_idx,                            # index_state
                False,                              # playing_state
                gr.update(value="▶"),           # play_pause_btn label
            )
        step_table.select(_on_step_select, inputs=[frames_state], outputs=[frame_image, frame_slider, index_state, playing_state, play_pause_btn])

        # Play/Pause handler
        def _on_play_pause(playing: bool):
            new_playing = not bool(playing)
            return new_playing, gr.update(value=("⏸" if new_playing else "▶"))

        def _on_slider_change(frames: List[np.ndarray], val: int, playing: bool):
            """Update current frame when user releases the slider.

            Preserve current playing state so programmatic slider updates during autoplay
            don't pause playback.
            """
            idx = int(val) if val is not None else 0
            img = frames[idx] if (isinstance(frames, list) and 0 <= idx < len(frames)) else None
            return gr.update(value=img), idx, playing, gr.update(value=("⏸" if playing else "▶"))
        play_pause_btn.click(_on_play_pause, inputs=[playing_state], outputs=[playing_state, play_pause_btn])
        # While dragging, update the frame live for fast visual scanning (and pause playback)
        def _on_slider_input(frames: List[np.ndarray], val: int | float | None):
            # Use the slider's current value passed as an input to avoid any evt.value staleness
            idx = int(val) if val is not None else 0
            img = frames[idx] if (isinstance(frames, list) and 0 <= idx < len(frames)) else None
            # Pause while scrubbing for smoother UX and to avoid race with autoplay
            return gr.update(value=img), idx, False, gr.update(value="▶")

        frame_slider.input(
            _on_slider_input,
            inputs=[frames_state, frame_slider],
            outputs=[frame_image, index_state, playing_state, play_pause_btn],
        )

        # Use release instead of change to avoid triggering on programmatic updates from the timer
        frame_slider.release(_on_slider_change, inputs=[frames_state, frame_slider, playing_state], outputs=[frame_image, index_state, playing_state, play_pause_btn])

        # Autoplay tick handler (only if timer available)
        def _on_tick(frames: List[np.ndarray], idx: int, playing: bool):
            if not playing or not frames:
                return gr.update(), gr.update(), idx, playing, gr.update()
            if int(idx) < len(frames) - 1:
                new_idx = int(idx) + 1
                return gr.update(value=frames[new_idx]), gr.update(value=new_idx), new_idx, True, gr.update(value="⏸")
            # Reached end: stop
            last_idx = len(frames) - 1
            return gr.update(value=frames[last_idx]), gr.update(value=last_idx), last_idx, False, gr.update(value="▶")

        if timer is not None:
            timer.tick(_on_tick, inputs=[frames_state, index_state, playing_state], outputs=[frame_image, frame_slider, index_state, playing_state, play_pause_btn])

        # CSV export handler
        def _export_csv(rows: List[List[Any]] | None, rid: str, ckpt_label: str | None):
            import csv
            import tempfile
            import os
            if not rows:
                # No rows to export; do nothing
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
