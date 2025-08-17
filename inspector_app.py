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

from play import load_model as _load_model
from play import load_config_from_run as _load_config_from_run


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
        if name == "best_checkpoint.ckpt":
            return (0, -p.stat().st_mtime)
        if name == "last_checkpoint.ckpt":
            return (1, -p.stat().st_mtime)
        return (2, -p.stat().st_mtime)

    files.sort(key=score)

    labels: List[str] = []
    mapping: Dict[str, Path] = {}
    for p in files:
        label = p.name
        if label == "best_checkpoint.ckpt":
            label = "best_checkpoint.ckpt (best)"
        elif label == "last_checkpoint.ckpt":
            label = "last_checkpoint.ckpt (last)"
        labels.append(label)
        mapping[label] = p

    default_label = next((l for l in labels if l.startswith("best_checkpoint")), labels[0])
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

    frames: List[np.ndarray] = []
    steps: List[Dict[str, Any]] = []

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    total_reward = 0.0
    t = 0

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
            t += 1

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

            if terminated or truncated:
                break
    finally:
        env.close()

    return frames, steps, {
        "total_reward": float(total_reward),
        "steps": t,
        "env_id": config.env_id,
        "algo_id": config.algo_id,
        "checkpoint_name": Path(ckpt_path).name,
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

        with gr.Row():
            frames_gallery = gr.Gallery(label="Frames", columns=4, height=400)
        with gr.Row():
            step_table = gr.Dataframe(
                headers=["step", "action", "reward", "cum_reward", "value", "done", "probs"],
                datatype=["number", "number", "number", "number", "number", "bool", "str"],
                row_count=(0, "dynamic"),
                col_count=(7, "fixed"),
                label="Per-step details",
            )
        with gr.Row():
            summary = gr.JSON(label="Summary")

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
                    s["step"], s["action"], s["reward"], s["cum_reward"], s["value"], s["done"],
                    "[" + ", ".join(f"{p:.3f}" for p in (s["probs"] or [])) + "]" if s["probs"] else None,
                ])
            return frames, rows, info

        run_btn.click(_inspect, inputs=[run_id, checkpoint, deterministic, max_steps], outputs=[frames_gallery, step_table, summary])

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
