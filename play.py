#!/usr/bin/env python3
"""
Alternate play script: minimal and reuse-first.

- Loads a policy using the existing `play.load_model` helper.
- Builds a single-env environment with `render_mode='human'`.
- Plays episodes by stepping via `utils.rollouts.RolloutCollector`.

Keeps this file tiny by importing helpers from the main play script
and utils, avoiding code duplication.
"""

from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path

from utils.environment import build_env
from utils.rollouts import RolloutCollector


def read_policy(run_id: str, model_path: str | None):
    """Return (policy_model, config, resolved_model_path).

    Uses the main play.py helpers to avoid duplicating checkpoint/config logic.
    """
    from play import load_model, load_config_from_run, find_best_checkpoint_in_run

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

