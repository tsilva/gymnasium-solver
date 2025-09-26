#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform

from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.rollouts import RolloutCollector
from utils.policy_ops import policy_act
from utils.torch import _device_of
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
    config = run.load_config()
    env = build_env_from_config(
        config,
        n_envs=1,
        subproc=False,
        render_mode="human" if not args.no_render else None
    )

    # Attach a live observation bar printer for interactive play (vector-level wrapper)
    from gym_wrappers.vec_obs_printer import VecObsBarPrinter
    env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=True, target_episodes=target_episodes)

    # Load configuration
    # TODO: we should be loading the agent and having it run the episode
    policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)

    # TODO: make step-by-step mode use the same rollout collector but just with n_steps=1, meaning we can drop most code below use the same code loop as when we dont have step-by-step mode
    # Step-by-step interactive mode for visual debugging
    if args.step_by_step:
        import torch

        print(f"Playing {target_episodes} episode(s) step-by-step with render_mode='human'...")

        device = _device_of(policy_model)
        obs = env.reset()
        reported_episodes = 0
        step = 0
        # Offer a tiny hint once
        print("Press Enter to step, 'q' then Enter to quit.")
        while reported_episodes < target_episodes:
            # Wait for user input before each step
            try:
                user = input("")
            except EOFError:
                user = ""
            if isinstance(user, str) and user.strip().lower() in {"q", "quit", "exit"}:
                break

            # Compute action from current observation
            obs_t = torch.as_tensor(obs, device=device)
            actions_t, _logps_t, _values_t = policy_act(policy_model, obs_t, deterministic=args.deterministic)
            actions_np = actions_t.detach().cpu().numpy()

            # Environment step (VecEnv auto-resets done envs)
            next_obs, rewards, dones, infos = env.step(actions_np)
            step += 1


            # Episode accounting based on VecEnv infos
            if hasattr(dones, "__len__") and len(dones) > 0 and bool(dones[0]):
                # Try to read episode summary from info when available
                ep_rew = None
                ep_len = None
                try:
                    info0 = infos[0]
                    ep = info0.get("episode") if isinstance(info0, dict) else None
                    if isinstance(ep, dict) and "r" in ep:
                        ep_rew = float(ep["r"])
                    if isinstance(ep, dict) and "l" in ep:
                        ep_len = int(ep["l"])
                except Exception:
                    ep_rew = ep_rew
                    ep_len = ep_len

                reported_episodes += 1
                if ep_rew is not None:
                    # Prefer episode summary from info when available
                    last_len = ep_len if ep_len is not None else "--"
                    print(
                        f"[episodes {reported_episodes}/{target_episodes}] last_rew={ep_rew:+.3f} last_len={last_len}"
                    )
                else:
                    print(f"[episodes {reported_episodes}/{target_episodes}] finished")

                # Advance observation
                obs = next_obs

                # Nothing else to reset here; VecObsBarPrinter tracks ep counters for header

        print("Done.")
        return

    # Default: continuous play using RolloutCollector
    # Initialize rollout collector with training-time hyperparams
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    # Initialize obs on first collect; keep collecting until target episodes reached
    print(f"Playing {target_episodes} episode(s) with render_mode='human'...")

    # Collect episodes until target episodes reached
    reported_episodes = 0
    while reported_episodes < target_episodes:
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
