#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform

from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.rollouts import RolloutCollector
from utils.run import Run


def play_episodes_manual(env, target_episodes: int, mode: str, step_by_step: bool = False):
    """Play episodes with random actions or user input."""
    import numpy as np

    # Get action space info
    action_space = env.single_action_space
    n_actions = action_space.n

    if mode == "user":
        print(f"\nAction controls: Press 0-{n_actions-1} to select action, Enter to execute.")
        if step_by_step:
            print("Step-by-step mode enabled. Press Enter to step, 'q' then Enter to quit.")
    else:
        print(f"\nRandom action mode enabled. Sampling from {n_actions} actions.")
        if step_by_step:
            print("Step-by-step mode enabled. Press Enter to step, 'q' then Enter to quit.")

    reported_episodes = 0
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0

    while reported_episodes < target_episodes:
        # Choose action based on mode
        if mode == "random":
            action = action_space.sample()
            if step_by_step:
                print(f"Random action: {action}")
        else:  # user mode
            if step_by_step:
                try:
                    user_input = input(f"[step {episode_length}] Action (0-{n_actions-1}): ")
                except EOFError:
                    user_input = ""

                if user_input.strip().lower() in {"q", "quit", "exit"}:
                    break

                try:
                    action = int(user_input.strip())
                    assert 0 <= action < n_actions, f"Action must be 0-{n_actions-1}"
                except (ValueError, AssertionError) as e:
                    print(f"Invalid action: {e}. Using action 0.")
                    action = 0
            else:
                # Non-step-by-step user mode: read single character
                try:
                    user_input = input()
                except EOFError:
                    user_input = ""

                if user_input.strip().lower() in {"q", "quit", "exit"}:
                    break

                try:
                    action = int(user_input.strip()[0]) if user_input.strip() else 0
                    action = max(0, min(action, n_actions - 1))
                except (ValueError, IndexError):
                    action = 0

        # Execute action
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        episode_reward += reward[0]
        episode_length += 1

        # Check if episode finished
        done = terminated[0] or truncated[0]
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            reported_episodes += 1

            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            print(
                f"[episodes {reported_episodes}/{target_episodes}] last_rew={episode_reward:.2f} last_len={episode_length} "
                f"mean_rew={mean_reward:.2f} mean_len={int(mean_length)}"
            )

            # Reset for next episode
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0


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
    p.add_argument(
        "--mode",
        choices=["trained", "random", "user"],
        default="trained",
        help="Action mode: 'trained' (use trained policy), 'random' (sample from action space), 'user' (keyboard input)",
    )
    args = p.parse_args()
    target_episodes = max(1, int(args.episodes))

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
    if is_wsl: os.environ.setdefault("SDL_RENDER_DRIVER", "software")

    # Load run and config
    run = Run.load(args.run_id)
    config = run.load_config()

    # For trained mode, require checkpoint
    if args.mode == "trained":
        assert run.best_checkpoint_dir is not None, "run has no best checkpoint"

    # Build a single-env environment with human rendering
    # Force vectorization_mode='sync' to ensure render() is supported (ALE native vectorization doesn't support it)
    env = build_env_from_config(
        config,
        n_envs=1,
        vectorization_mode='sync',
        render_mode="human" if not args.no_render else None
    )

    # Attach a live observation bar printer for interactive play (vector-level wrapper)
    from gym_wrappers.vec_obs_printer import VecObsBarPrinter
    env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=True, target_episodes=target_episodes)

    # Handle different modes
    if args.mode in ["random", "user"]:
        # Manual control modes don't need policy
        play_episodes_manual(env, target_episodes, args.mode, args.step_by_step)
        print("Done.")
        return

    # Trained mode: load policy
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
