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
    # Default to '@last' which resolves to the most recently created run.
    # If that run has no best checkpoint yet, we fall back to the most recent
    # run that does have a best checkpoint.
    p.add_argument("--run-id", default="@last", help="Run ID under runs/ (default: @last; falls back to latest run with a best checkpoint)")  # TODO: remove @last hardcode (this is a Run SOC)
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions (mode/argmax)")
    args = p.parse_args()

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
    if is_wsl: os.environ.setdefault("SDL_RENDER_DRIVER", "software")

    # Load run
    run = Run.load(args.run_id)
    assert run.best_checkpoint_dir is not None, "run has no best checkpoint"

    # Build a single-env environment with human rendering
    config = run.load_config()
    env = build_env_from_config(
        config,
        n_envs=1,
        subproc=False,
        render_mode="human" # TODO: force n_envs when render_mode is human
    )

    # Attach a live observation bar printer for interactive play (vector-level wrapper)
    try:
        from gym_wrappers.vec_obs_printer import VecObsBarPrinter
        env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=True, target_episodes=args.episodes)
    except Exception:
        # Non-fatal: continue without the visualizer if the wrapper fails to attach
        pass

    # Resolve checkpoint path with graceful fallbacks
    from pathlib import Path

    def _pick_latest_best_ckpt() -> tuple[Path | None, str | None]:
        runs_root = Path("runs")
        candidates: list[Path] = []
        for p in runs_root.glob("*/checkpoints/@best/policy.ckpt"):
            try:
                if p.is_file():
                    candidates.append(p)
            except FileNotFoundError:
                # Broken symlink or race; ignore
                pass
        if not candidates:
            return None, None
        best = max(candidates, key=lambda p: p.stat().st_mtime)
        return best, best.parent.parent.parent.name  # runs/<id>/checkpoints/@best/policy.ckpt

    ckpt_path = run.best_checkpoint_dir / "policy.ckpt"
    if not ckpt_path.exists():
        # Try last checkpoint within the same run first
        candidate = run.last_checkpoint_dir / "policy.ckpt"
        if candidate.exists():
            ckpt_path = candidate
        else:
            # If the user asked for '@last', pick the latest run that has a best checkpoint
            if args.run_id == "@last":
                fallback_ckpt, fallback_run_id = _pick_latest_best_ckpt()
                if fallback_ckpt is not None and fallback_run_id is not None:
                    print(f"Best checkpoint not found in last run; falling back to runs/{fallback_run_id}.")
                    run = Run.load(fallback_run_id)
                    config = run.load_config()
                    ckpt_path = fallback_ckpt
                else:
                    raise FileNotFoundError(
                        "No checkpoints found. Consider training a model or passing --run-id."
                    )
            else:
                raise FileNotFoundError(
                    f"Best checkpoint not found for run '{run.id}' at {ckpt_path}."
                )

    # Load configuration
    # TODO: we should be loading the agent and having it run the episode
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
