#!/usr/bin/env python3
"""
Alternate play script: minimal and reuse-first.

- Loads a policy from a run's best/last checkpoint.
- Builds a single-env environment with `render_mode='human'`.
- Plays episodes by stepping via `utils.rollouts.RolloutCollector`.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

from utils.environment import build_env
from utils.rollouts import RolloutCollector
from typing import Any

# Local constants
RUNS_DIR = Path("runs")


def _resolve_run_dir(run_id: str) -> Path:
    """Resolve a run directory from an id or special alias.

    - Accepts explicit paths, a bare run id under `runs/`, or the alias
      `@latest-run` (with legacy `latest-run` fallback).
    """
    p = Path(run_id)
    if p.exists():
        return p
    if run_id in {"@latest-run", "latest-run"}:
        latest = RUNS_DIR / "@latest-run"
        if not latest.exists():
            legacy = RUNS_DIR / "latest-run"
            latest = legacy if legacy.exists() else latest
        return latest
    return RUNS_DIR / run_id


def load_config_from_run(run_id: str):
    """Load a run's config.json as a utils.config.Config instance.

    Falls back from runs/<id>/config.json to runs/<id>/configs/config.json.
    """
    import json
    from utils.config import Config

    run_dir = _resolve_run_dir(run_id)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        alt = run_dir / "configs" / "config.json"
        if alt.exists():
            cfg_path = alt
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found under run: {run_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    algo_id = str(data.get("algo_id", "")).lower()
    if not algo_id:
        raise RuntimeError("Invalid config.json: missing algo_id")
    return Config.create_for_algo(algo_id, data)


def find_best_checkpoint_in_run(run_id: str) -> Path:
    """Return a representative checkpoint path from a run directory.

    Preference order: best.ckpt -> last.ckpt -> most recent *.ckpt.
    """
    run_dir = _resolve_run_dir(run_id)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints/ directory under run: {run_dir}")

    for name in ("best.ckpt", "last.ckpt", "best.ckpt", "last.ckpt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    files = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if files:
        return files[0]
    raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")


def load_model(ckpt_path: Path, config):
    """Build a policy model from config/env shapes and load weights from a checkpoint.

    Creates a short-lived helper env to infer input/output shapes, then closes it
    before constructing the long-lived env used for rendering.
    """
    import torch
    from utils.environment import build_env as _build_env
    from utils.policy_factory import create_actor_critic_policy, create_policy

    # Helper env strictly for shape inference
    helper_env = _build_env(
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
        obs_space = getattr(helper_env, "observation_space", None)
        act_space = getattr(helper_env, "action_space", None)
        input_shape = None
        output_shape = None
        if obs_space is not None and hasattr(obs_space, "shape") and obs_space.shape:
            input_shape = tuple(int(s) for s in obs_space.shape)
        elif input_dim is not None:
            input_shape = (int(input_dim),)

        if act_space is not None and hasattr(act_space, "n"):
            output_shape = (int(act_space.n),)
        elif output_dim is not None:
            output_shape = (int(output_dim),)

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


def read_policy(run_id: str, model_path: str | None):
    """Return (policy_model, config, resolved_model_path).

    Uses local helpers to load the run config and resolve a checkpoint path.
    """
    # Load config from the run directory
    config = load_config_from_run(run_id)

    # Resolve checkpoint path
    ckpt_path = Path(model_path) if model_path else find_best_checkpoint_in_run(run_id)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model file not found: {ckpt_path}")

    # Build policy according to config and load weights
    policy_model = load_model(ckpt_path, config)
    return policy_model, config, ckpt_path


def main():
    p = argparse.ArgumentParser(description="Play a trained agent using RolloutCollector (human render)")
    p.add_argument("--run-id", default="@latest-run", help="Run ID to load (defaults to @latest-run)")
    p.add_argument("--model", default=None, help="Optional explicit checkpoint path (overrides --run-id)")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions (mode/argmax)")
    args = p.parse_args()

    # Load policy and config
    policy_model, config, ckpt_path = read_policy(args.run_id, args.model)
    print(f"Using checkpoint: {ckpt_path}")

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    try:
        is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
        if is_wsl:
            os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    except Exception:
        pass

    # Build a single-env environment with human rendering
    env = build_env(
        config.env_id,
        seed=config.seed,
        n_envs=1,
        subproc=False,
        env_wrappers=config.env_wrappers,
        norm_obs=config.normalize_obs,
        frame_stack=config.frame_stack,
        obs_type=config.obs_type,
        render_mode="human",
        env_kwargs=config.env_kwargs,
        grayscale_obs=getattr(config, "grayscale_obs", False),
        resize_obs=getattr(config, "resize_obs", False),
    )

    try:
        # Initialize rollout collector with training-time hyperparams
        n_steps = int(config.n_steps) if getattr(config, "n_steps", None) else 2048
        collector = RolloutCollector(
            env=env,
            policy_model=policy_model,
            n_steps=n_steps,
            **config.rollout_collector_hyperparams(),
        )

        # Initialize obs on first collect; keep collecting until target episodes reached
        target_eps = max(1, int(args.episodes))
        start_eps = collector.total_episodes
        print(f"Playing {target_eps} episode(s) with render_mode='human'...")

        while (collector.total_episodes - start_eps) < target_eps:
            _ = collector.collect(deterministic=args.deterministic)
            m = collector.get_metrics()
            played = collector.total_episodes - start_eps
            print(
                f"[episodes {played}/{target_eps}] last_rew={m.get('ep_rew_last', 0):.2f} "
                f"mean_rew={m.get('ep_rew_mean', 0):.2f} fps={m.get('rollout_fps', 0):.1f}"
            )

        print("Done.")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
