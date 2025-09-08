#!/usr/bin/env python3
"""
Inspector application logic.

Run via: python inspector.py [args]
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

# UI constants to keep strings consistent
DISPLAY_RAW = "Rendered (raw)"
DISPLAY_STACK = "Processed (stack)"
FRAME_LABEL_RAW = "Frame (raw)"
FRAME_LABEL_STACK = "Frame (processed stack)"
PLAY_ICON = "\u25B6"  # ▶
PAUSE_ICON = "\u23F8"  # ⏸

# Local minimal loaders to avoid depending on play.py helpers
def _load_config_from_run(run_id: str):
    """Load a run's config.json as an attribute-access object.

    Falls back from runs/<id>/config.json to runs/<id>/configs/config.json.
    Returns a simple object exposing keys as attributes, with sensible defaults.
    """
    import json
    from types import SimpleNamespace

    run_dir = _resolve_run_dir(run_id)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        alt = run_dir / "configs" / "config.json"
        if alt.exists():
            cfg_path = alt
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found under run: {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Provide defaults to fields the inspector uses
    defaults = {
        "env_wrappers": [],
        "normalize_obs": False,
        "frame_stack": 1,
        "obs_type": "rgb",
        "env_kwargs": {},
        "grayscale_obs": False,
        "resize_obs": False,
        "policy_kwargs": {},
        "policy": "mlp",
        "hidden_dims": (64, 64),
        "activation": "relu",
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }
    for k, v in defaults.items():
        data.setdefault(k, v)

    return SimpleNamespace(**data)


def _load_model(ckpt_path: Path, config):
    """Build a policy model from config/env shapes and load weights from a checkpoint.

    Creates a short-lived helper env to infer input/output shapes, then closes it
    before the long-lived env used by the inspector is constructed (important for
    Retro environments that do not allow multiple emulator instances per process).
    """
    from utils.environment import build_env
    from utils.policy_factory import create_actor_critic_policy, create_policy

    # Helper env strictly for shape inference
    helper_env = build_env(
        config.env_id,
        seed=getattr(config, "seed", 42),
        env_wrappers=getattr(config, "env_wrappers", []),
        norm_obs=getattr(config, "normalize_obs", False),
        n_envs=1,
        frame_stack=getattr(config, "frame_stack", 1),
        obs_type=getattr(config, "obs_type", None),
        render_mode=None,
        env_kwargs=getattr(config, "env_kwargs", {}),
        subproc=False,
        grayscale_obs=getattr(config, "grayscale_obs", False),
        resize_obs=getattr(config, "resize_obs", False),
    )

    try:
        input_shape = helper_env.observation_space
        output_shape = helper_env.action_space
        if not input_shape or not output_shape:
            raise RuntimeError("Could not infer model input/output shapes from environment")

        policy_type = str(getattr(config, "policy", "mlp")).lower()
        hidden_dims = getattr(config, "hidden_dims", (64, 64))
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        activation = str(getattr(config, "activation", "relu"))
        policy_kwargs = getattr(config, "policy_kwargs", {}) or {}

        algo_id = str(getattr(config, "algo_id", "")).lower()
        if algo_id == "ppo":
            model = create_actor_critic_policy(
                policy_type,
                input_shape=input_shape,
                output_shape=output_shape,
                hidden_dims=hidden_dims,
                activation=activation,
                **policy_kwargs,
            )
        else:
            # Policy-only (REINFORCE and other stateless baselines)
            model = create_policy(
                policy_type,
                input_shape=input_shape,
                output_shape=output_shape,
                hidden_dims=hidden_dims,
                activation=activation,
                **policy_kwargs,
            )

        # Load weights
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Invalid checkpoint: missing model_state_dict in {ckpt_path}")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    finally:
        try:
            helper_env.close()
        except Exception:
            pass
from utils.rollouts import (
    compute_batched_mc_returns,
    compute_batched_gae_advantages_and_returns,
)


def compute_mc_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted Monte Carlo returns for a single trajectory.

    Delegates to the batched implementation with batch size 1 for consistency
    with training-time logic.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    rewards_b = rewards.reshape(-1, 1)
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
    """Compute GAE(λ) advantages and returns for a single (T,) trajectory.

    Uses the batched implementation with batch size 1 to mirror collector behavior.
    """
    values_b = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    rewards_b = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
    dones_b = np.asarray(dones, dtype=bool).reshape(-1, 1)
    if timeouts is None:
        timeouts_b = np.zeros_like(dones_b, dtype=bool)
    else:
        timeouts_b = np.asarray(timeouts, dtype=bool).reshape(-1, 1)

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


RUNS_DIR = Path("runs")


def _resolve_run_dir(run_id: str) -> Path:
    if run_id in {"latest-run", "@latest-run"}:
        # Prefer new '@latest-run' symlink, fallback to legacy name
        latest = RUNS_DIR / "@latest-run"
        if not latest.is_symlink():
            legacy = RUNS_DIR / "latest-run"
            latest = legacy if legacy.is_symlink() else latest
        if latest.is_symlink():
            run_id = str(latest.readlink())
        else:
            raise FileNotFoundError("@latest-run symlink not found")
    run_path = RUNS_DIR / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    return run_path


def list_runs() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [
        p.name
        for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name not in {"@latest-run", "latest-run"}
    ]
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
        # Prefer best, then last, then everything else by recency.
        if name in {"best.ckpt", "best_checkpoint.ckpt"}:  # support legacy and new names
            return (0, -p.stat().st_mtime)
        if name in {"last.ckpt", "last_checkpoint.ckpt"}:
            return (1, -p.stat().st_mtime)
        return (2, -p.stat().st_mtime)

    files.sort(key=score)

    labels: List[str] = []
    mapping: Dict[str, Path] = {}
    for p in files:
        label = p.name
        if label in {"best.ckpt", "best_checkpoint.ckpt"}:
            label = f"{label} (best)"
        elif label in {"last.ckpt", "last_checkpoint.ckpt"}:
            label = f"{label} (last)"
        labels.append(label)
        mapping[label] = p

    default_label = next((l for l in labels if l.startswith("best")), labels[0])
    return labels, mapping, default_label


# Note: Action labels/spec/fps fallbacks are provided by VecInfoWrapper methods.
# Avoid duplicating YAML parsing here; call env.get_action_labels(), env.get_spec(), etc.


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

    ckpt_labels, mapping, default_label = list_checkpoints_for_run(run_id)
    if not ckpt_labels:
        raise FileNotFoundError("No checkpoints found for this run")
    selected_label = checkpoint_label or default_label
    ckpt_path = mapping[selected_label]

    # Important for Retro environments: load the model (which briefly creates
    # a helper env to infer shapes) BEFORE creating the main episode env.
    # Retro does not allow multiple emulator instances per process, so this
    # ordering ensures the helper env is closed prior to constructing the
    # long-lived env used for stepping/recording here.
    policy_model = _load_model(ckpt_path, config)
    policy_model.eval()

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

    # Load action labels from vec env wrapper if available
    action_labels: List[str] | None = None
    try:
        if hasattr(env, "get_action_labels"):
            act_labels_raw = env.get_action_labels()  # type: ignore[attr-defined]
            if isinstance(act_labels_raw, list):
                action_labels = [str(x) for x in act_labels_raw]
    except Exception:
        action_labels = None

    # Collect environment spec for summary tabs (VecInfoWrapper exposes safe helpers)
    env_spec_obj = env.get_spec() if hasattr(env, "get_spec") else None
    reward_range = env.get_reward_range() if hasattr(env, "get_reward_range") else None
    reward_threshold = env.get_reward_threshold() if hasattr(env, "get_reward_threshold") else None
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
        "spec_id": (str(getattr(env_spec_obj, "id", None)) if env_spec_obj is not None else None),
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

    # No single-frame processed RGB conversion needed; UI focuses on raw render and stack grid.

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

    def _vector_stack_to_frames(vec: np.ndarray, n_stack_hint: int | None, row_height: int = 8) -> List[np.ndarray]:
        """Represent a stacked 1D observation as a list of grayscale bar images.

        Splits the vector into ``n_stack_hint`` equal chunks (when possible) and
        turns each chunk into an ``(row_height, chunk_len, 1)`` image with values
        normalized per-chunk to [0, 1] for visibility. Falls back to a single
        bar if splitting is not possible or the hint is invalid.
        """
        try:
            v = np.asarray(vec).reshape(-1)
            n = int(n_stack_hint) if (n_stack_hint is not None) else 1
            if n <= 1:
                n = 1
            L = v.shape[0]
            frames: List[np.ndarray] = []
            if n > 1 and L % n == 0:
                seg = L // n
                chunks = [v[i * seg:(i + 1) * seg] for i in range(n)]
            else:
                chunks = [v]
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
        except Exception:
            return []
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
    # Derive a fixed stacking hint once
    try:
        n_stack_hint = int(getattr(config, "frame_stack", None) or 0)
    except Exception:
        n_stack_hint = None
    has_stack_hint = bool(n_stack_hint is not None and int(n_stack_hint) > 1)

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
                stack_img = None
                # n_stack_hint precomputed above

                # Build a stack grid from the observation channels when frame stacking is enabled
                if isinstance(obs0, np.ndarray) and obs0.ndim in (2, 3):
                    hwc = _ensure_hwc(obs0)
                    if has_stack_hint:
                        split = _split_stack(hwc, assume_rgb_groups=True, n_stack_hint=n_stack_hint)
                        if isinstance(split, list) and len(split) >= 2:
                            grid = _make_grid(split, cols=None)
                            if grid is not None:
                                stack_img = grid
                # Vector observation fallback (only if we still don't have a frame-based stack)
                elif stack_img is None and isinstance(obs0, np.ndarray) and (obs0.ndim == 1 or (obs0.ndim == 2 and 1 in obs0.shape)) and obs_type_cfg not in {"ram", "objects"}:
                    if has_stack_hint:
                        vec = obs0.reshape(-1)
                        frames_vec = _vector_stack_to_frames(vec, n_stack_hint=n_stack_hint, row_height=8)
                        if frames_vec:
                            grid = _make_grid(frames_vec, cols=None)
                            if grid is not None:
                                stack_img = grid

                # As a final fallback for non-image observations, build a grid from the last N rendered frames
                if stack_img is None and has_stack_hint and isinstance(frames, list) and len(frames) >= 1:
                    last = frames[-int(n_stack_hint):]
                    grid_from_frames = _make_grid([_ensure_hwc(f) for f in last], cols=None)
                    if grid_from_frames is not None:
                        stack_img = grid_from_frames

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

            action_idx = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            steps.append({
                "step": t,
                "action": action_idx,
                "action_label": (action_labels[action_idx] if action_labels is not None and 0 <= action_idx < len(action_labels) else None),
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

    # Only return stack frames if lists are non-empty (else None)
    stack_out = stack_frames if stack_frames else None

    return frames, None, stack_out, steps, {
        "total_reward": float(total_reward),
        "steps": t,
        "env_id": config.env_id,
        "algo_id": config.algo_id,
        "checkpoint_name": Path(ckpt_path).name,
        "env_spec": env_spec_summary,
        "model_spec": model_spec_summary,
        "checkpoint_metrics": checkpoint_metrics_summary,
    }


def build_ui(default_run_id: str = "@latest-run"):
    import gradio as gr

    runs = list_runs()
    initial_run = default_run_id if default_run_id else (runs[0] if runs else "@latest-run")
    labels, mapping, default_label = (
        list_checkpoints_for_run(initial_run)
        if (runs or initial_run in {"@latest-run", "latest-run"})
        else ([], {}, None)
    )

    with gr.Blocks(theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # Run Inspector
        Select a run and checkpoint to visualize an episode: frames, actions, rewards, values.
        """)

        with gr.Row():
            run_id = gr.Dropdown(
                label="Run ID",
                choices=["@latest-run"] + runs,
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
                    choices=[DISPLAY_RAW],
                    value=DISPLAY_RAW,
                    interactive=True,
                )
                frame_image = gr.Image(label=FRAME_LABEL_RAW, height=400, type="numpy", image_mode="RGB")
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
        try:
            TimerCls = getattr(gr, "Timer", None)
            if TimerCls is not None:
                timer = TimerCls(1/30.0)  # default to 30 FPS
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
            try:
                if steps and 0 <= int(idx) < len(steps):
                    return _verticalize_row(_row_vals_from_step_dict(steps[int(idx)]))
            except Exception:
                pass
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

        def _inspect(rid: str, ckpt_label: str | None, det: bool, nsteps: int):
            frames_raw, _frames_proc_unused, frames_stack, steps, info = run_episode(rid, ckpt_label, det, int(nsteps))
            def _round3(x):
                try:
                    return round(float(x), 3)
                except Exception:
                    return x
            def _format_probs(probs: Any) -> str | None:
                try:
                    if isinstance(probs, list):
                        return "[" + ", ".join(f"{float(p):.3f}" for p in probs) + "]"
                except Exception:
                    pass
                return str(probs) if probs is not None else None
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
            rows = [_table_row_from_step(s) for s in steps]
            # Initialize gallery selection, states, play button, and slider range
            # Compute available display modes
            has_stack = isinstance(frames_stack, list) and len(frames_stack) > 0
            choices = [DISPLAY_RAW] + ([DISPLAY_STACK] if has_stack else [])

            if has_stack:
                display_value = DISPLAY_STACK
                frame_label = FRAME_LABEL_STACK
                active_frames = frames_stack
            else:
                display_value = DISPLAY_RAW
                frame_label = FRAME_LABEL_RAW
                active_frames = frames_raw

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

        def _on_slider_change(frames: List[np.ndarray], val: int, playing: bool, steps: List[Dict[str, Any]] | None):
            """Update current frame when user releases the slider, preserving play state."""
            img, row_val, i = _current_view_updates(frames, val, steps)
            return gr.update(value=img), gr.update(value=row_val), i, playing, gr.update(value=(PAUSE_ICON if playing else PLAY_ICON))
        play_pause_btn.click(_on_play_pause, inputs=[playing_state], outputs=[playing_state, play_pause_btn])
        # While dragging, update the frame live for fast visual scanning (and pause playback)
        def _on_slider_input(frames: List[np.ndarray], val: int | float | None, steps: List[Dict[str, Any]] | None):
            # Pause while scrubbing for smoother UX and to avoid race with autoplay
            img, row_val, i = _current_view_updates(frames, val, steps)
            return gr.update(value=img), gr.update(value=row_val), i, False, gr.update(value=PLAY_ICON)

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
                img, row_val, _ = _current_view_updates(frames, new_idx, steps)
                return gr.update(value=img), gr.update(value=new_idx), gr.update(value=row_val), new_idx, True, gr.update(value=PAUSE_ICON)
            # Reached end: stop
            last_idx = len(frames) - 1
            img, row_val, _ = _current_view_updates(frames, last_idx, steps)
            return gr.update(value=img), gr.update(value=last_idx), gr.update(value=row_val), last_idx, False, gr.update(value=PLAY_ICON)

        if timer is not None:
            timer.tick(_on_tick, inputs=[frames_state, index_state, playing_state, steps_state], outputs=[frame_image, frame_slider, current_step_table, index_state, playing_state, play_pause_btn])

        # Display mode switcher
        def _on_display_mode(mode: str, raw: List[np.ndarray], stack: List[np.ndarray], steps: List[Dict[str, Any]] | None):
            mode = str(mode or DISPLAY_RAW)
            if mode == DISPLAY_STACK and isinstance(stack, list) and len(stack) > 0:
                active = stack
                label = FRAME_LABEL_STACK
            else:
                active = raw if isinstance(raw, list) else []
                label = FRAME_LABEL_RAW
            return _initial_display_state(active, label, steps)

        display_mode.change(
            _on_display_mode,
            inputs=[display_mode, frames_raw_state, frames_stack_state, steps_state],
            outputs=[frame_image, frame_slider, frames_state, index_state, playing_state, play_pause_btn, current_step_table],
        )

        # CSV export handler
        def _export_csv(rows: List[List[Any]] | None, rid: str, ckpt_label: str | None):
            import csv
            import tempfile

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
    parser.add_argument("--run-id", type=str, default="@latest-run", help="Run ID under runs/ (default: @latest-run)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    args = parser.parse_args()

    demo = build_ui(args.run_id)
    demo.launch(server_port=args.port, server_name=args.host, share=args.share)


if __name__ == "__main__":
    main()
