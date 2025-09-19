#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform

from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.rollouts import RolloutCollector
from utils.run import Run


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

    # Load run
    run = Run.load(args.run_id)
    assert run.best_checkpoint_path is not None, "run has no best checkpoint"

    # Build a single-env environment with human rendering
    config = run.load_config()
    env = build_env_from_config(
        config,
        n_envs=1,
        subproc=False,
        render_mode="human" # TODO: force n_envs when render_mode is human
    )

    # Load configuration 
    policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)

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
    print(f"Playing {target_eps} episode(s) with render_mode='human'...")

    # Collect episodes until target episodes reached
    reported_episodes = 0
    while reported_episodes < target_eps:
        _ = collector.collect(deterministic=args.deterministic)
        finished_eps = collector.pop_recent_episodes()
        if not finished_eps:
            continue  # Keep collecting until we finish a full episode

        for _env_idx, ep_rew, _ep_len, _was_timeout in finished_eps:
            if reported_episodes >= target_eps:
                break

            reported_episodes += 1
            metrics = collector.get_metrics()
            mean_rew = metrics.get('ep_rew_mean', 0)
            fps = metrics.get('rollout_fps', 0)
            print(
                f"[episodes {reported_episodes}/{target_eps}] last_rew={ep_rew:.2f} "
                f"mean_rew={mean_rew:.2f} fps={fps:.1f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
