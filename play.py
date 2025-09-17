#!/usr/bin/env python3

from __future__ import annotations

import os
import platform
import argparse
from pathlib import Path

from utils.environment import build_env_from_config
from utils.rollouts import RolloutCollector
from utils.run import Run
from utils.policy_factory import load_policy_model_from_checkpoint

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Play a trained agent using RolloutCollector (human render)")
    p.add_argument("--run-id", default="@latest-run", help="Run ID to load (defaults to @latest-run)")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions (mode/argmax)")
    args = p.parse_args()

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
    if is_wsl: os.environ.setdefault("SDL_RENDER_DRIVER", "software")

    # Load checkpoint
    run = Run.from_id(args.run_id)
    ckpt_path = run.best_checkpoint_path
    config = run.load_config()
    print(f"Using checkpoint: {ckpt_path}")

    # Build a single-env environment with human rendering
    env = build_env_from_config(
        config,
        n_envs=1,
        subproc=False,
        render_mode="human" # TODO: force n_envs when render_mode is human
    )

    # Load configuration 
    policy_model, _ = load_policy_model_from_checkpoint(ckpt_path, env, config)

    # Initialize rollout collector with training-time hyperparams
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=config.n_steps,
        **config.rollout_collector_hyperparams(),
    )
    
    # TODO: do we need this extra collect?
    # Initialize obs on first collect; keep collecting until target episodes reached
    target_eps = max(1, int(args.episodes))
    start_eps = collector.total_episodes
    print(f"Playing {target_eps} episode(s) with render_mode='human'...")

    # Collect episodes until target episodes reached
    while (collector.total_episodes - start_eps) < target_eps:
        _ = collector.collect(deterministic=args.deterministic)
        m = collector.get_metrics()
        played = collector.total_episodes - start_eps
        played = min(played, target_eps) # NOTE: collector may collect more episodes than requested
        print(
            f"[episodes {played}/{target_eps}] last_rew={m.get('ep_rew_last', 0):.2f} "
            f"mean_rew={m.get('ep_rew_mean', 0):.2f} fps={m.get('rollout_fps', 0):.1f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
