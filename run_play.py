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
    p.add_argument("--run-id", default="@last", help="Run ID under runs/ (default: last run with best checkpoint)")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    p.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions (mode/argmax)")
    p.add_argument("--no-render", action="store_true", default=False, help="Do not render the environment")
    p.add_argument(
        "--step-by-step",
        dest="step_by_step",
        action="store_true",
        help="Pause for input each env step for visual debugging",
    )
    args = p.parse_args()
    target_episodes = max(1, int(args.episodes))

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
    if is_wsl: os.environ.setdefault("SDL_RENDER_DRIVER", "software")

    # Load run
    run = Run.load(args.run_id)
    assert run.best_checkpoint_dir is not None, "run has no best checkpoint"

    # Build a single-env environment with human rendering
    # Force vectorization_mode='sync' to ensure render() is supported (ALE native vectorization doesn't support it)
    config = run.load_config()
    env = build_env_from_config(
        config,
        n_envs=1,
        subproc=False,
        vectorization_mode='sync',
        render_mode="human" if not args.no_render else None
    )

    # Attach a live observation bar printer for interactive play (vector-level wrapper)
    from gym_wrappers.vec_obs_printer import VecObsBarPrinter
    env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=True, target_episodes=target_episodes)

    # Load configuration
    # TODO: we should be loading the agent and having it run the episode
    policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)

    # Initialize rollout collector; step-by-step mode uses single-step rollouts
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=1 if args.step_by_step else config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    if args.step_by_step:
        print(f"Playing {target_episodes} episode(s) step-by-step with render_mode='human'...")
        print("Press Enter to step, 'q' then Enter to quit.")
    else:
        print(f"Playing {target_episodes} episode(s) with render_mode='human'...")

    # Collect episodes until target episodes reached
    reported_episodes = 0
    while reported_episodes < target_episodes:
        if args.step_by_step:
            try:
                user = input("")
            except EOFError:
                user = ""
            if isinstance(user, str) and user.strip().lower() in {"q", "quit", "exit"}:
                break

        _ = collector.collect(deterministic=args.deterministic)
        finished_eps = collector.pop_recent_episodes()
        if not finished_eps:
            continue  # Keep collecting until we finish a full episode

        for _env_idx, ep_rew, _ep_len, _was_timeout in finished_eps:
            if reported_episodes >= target_episodes:
                break

            reported_episodes += 1
            metrics = collector.get_metrics()
            mean_rew = metrics.get('roll/ep_rew/mean', 0)
            last_len = int(_ep_len)
            mean_len = metrics.get('roll/ep_len/mean', 0)
            fps = metrics.get('roll/fps', 0)
            print(
                f"[episodes {reported_episodes}/{target_episodes}] last_rew={ep_rew:.2f} last_len={last_len} "
                f"mean_rew={mean_rew:.2f} mean_len={int(mean_len)} fps={fps:.1f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
