#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import platform
import time

from utils.environment import build_env_from_config
from utils.policy_factory import load_policy_model_from_checkpoint
from utils.random import set_random_seed
from utils.rollouts import RolloutCollector
from utils.run import Run


def play_episodes_manual(env, target_episodes: int, mode: str, step_by_step: bool = False, fps: int | None = None):
    """Play episodes with random actions or user input."""
    import numpy as np

    # Calculate frame delay for FPS limiting
    frame_delay = 1.0 / fps if fps else 0

    # Get action space info
    action_space = env.single_action_space
    n_actions = action_space.n

    # Try to import pygame for non-blocking keyboard input in GUI mode
    pygame_available = False
    pygame_screen = None
    if mode == "user" and not step_by_step:
        try:
            import pygame
            pygame.init()
            # Create a small control window to capture keyboard events
            pygame_screen = pygame.display.set_mode((400, 100))
            pygame.display.set_caption("Keyboard Controls - Press number keys (0-9)")
            pygame_available = True

            # Draw instructions on the control window
            font = pygame.font.Font(None, 24)
            pygame_screen.fill((40, 40, 40))
            text1 = font.render(f"Press keys 0-{n_actions-1} to select action", True, (255, 255, 255))
            text2 = font.render("Press Q to quit", True, (255, 255, 255))
            pygame_screen.blit(text1, (10, 25))
            pygame_screen.blit(text2, (10, 55))
            pygame.display.flip()
        except ImportError:
            pass

    if mode == "user":
        if step_by_step:
            print(f"\nAction controls: Press 0-{n_actions-1} to select action, Enter to execute.")
            print("Step-by-step mode enabled. Press Enter to step, 'q' then Enter to quit.")
        elif pygame_available:
            print(f"\nKeyboard control window opened!")
            print(f"Action controls: Press 0-{n_actions-1} keys in the control window to select action.")
            print("Press Q to quit.")
        else:
            print(f"\nAction controls: Press 0-{n_actions-1} to select action, Enter to execute.")
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
    action = 0  # Default action

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
            elif pygame_available:
                # Non-blocking pygame event handling
                action_changed = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if pygame_screen:
                            pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            if pygame_screen:
                                pygame.quit()
                            return
                        # Check for number keys (both main keyboard and numpad)
                        elif pygame.K_0 <= event.key <= pygame.K_9:
                            key_action = event.key - pygame.K_0
                            if 0 <= key_action < n_actions:
                                action = key_action
                                action_changed = True
                        elif pygame.K_KP0 <= event.key <= pygame.K_KP9:
                            key_action = event.key - pygame.K_KP0
                            if 0 <= key_action < n_actions:
                                action = key_action
                                action_changed = True

                # Update control window display to show current action
                if action_changed and pygame_screen:
                    font = pygame.font.Font(None, 24)
                    pygame_screen.fill((40, 40, 40))
                    text1 = font.render(f"Press keys 0-{n_actions-1} to select action", True, (255, 255, 255))
                    text2 = font.render("Press Q to quit", True, (255, 255, 255))
                    text3 = font.render(f"Current action: {action}", True, (100, 255, 100))
                    pygame_screen.blit(text1, (10, 15))
                    pygame_screen.blit(text2, (10, 45))
                    pygame_screen.blit(text3, (10, 75))
                    pygame.display.flip()
            else:
                # Fallback: blocking terminal input
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
        step_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        episode_reward += reward[0]
        episode_length += 1

        # Apply FPS limiting if specified
        if frame_delay > 0:
            elapsed = time.perf_counter() - step_start
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

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

    # Cleanup pygame if it was initialized
    if pygame_available and pygame_screen:
        import pygame
        pygame.quit()


def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Play a trained agent using RolloutCollector (human render)")
    id_group = p.add_mutually_exclusive_group(required=False)
    id_group.add_argument("--run-id", default=None, help="Run ID under runs/ (default: last run with best checkpoint)")
    id_group.add_argument("--config-id", default=None, help="Config ID in 'env:variant' format (e.g., 'VizDoom-Basic-v0:ppo') - runs with random/user policy")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    p.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions (mode/argmax)")
    p.add_argument("--headless", action="store_true", default=False, help="Do not render the environment")
    p.add_argument(
        "--step-by-step",
        dest="step_by_step",
        action="store_true",
        help="Pause for input each env step for visual debugging",
    )
    p.add_argument(
        "--mode",
        choices=["trained", "random", "user"],
        default=None,
        help="Action mode: 'trained' (use trained policy), 'random' (sample from action space), 'user' (keyboard input). Default: 'trained' for --run-id, 'random' for --config-id",
    )
    p.add_argument("--seed", type=str, default=None, help="Random seed for environment (int, 'train', 'val', 'test', or None for test seed)")
    p.add_argument("--fps", type=int, default=None, help="Limit playback to target FPS (frames per second)")
    args = p.parse_args()
    target_episodes = max(1, int(args.episodes))

    # Default to run-id mode if neither specified
    if args.run_id is None and args.config_id is None:
        args.run_id = "@best"

    # Best-effort: prefer software renderer on WSL to avoid GLX issues
    is_wsl = ("microsoft" in platform.release().lower()) or ("WSL_INTEROP" in os.environ)
    if is_wsl: os.environ.setdefault("SDL_RENDER_DRIVER", "software")

    # Branch: config-id mode or run-id mode
    if args.config_id is not None:
        # Config-id mode: load config from YAML, no trained policy
        from utils.config import load_config

        # Parse config_id in 'env:variant' format
        if ':' not in args.config_id:
            raise ValueError(f"config_id must be in 'env:variant' format (e.g., 'VizDoom-Basic-v0:ppo'), got: {args.config_id}")
        env_id, variant_id = args.config_id.split(':', 1)

        # Load config
        config = load_config(env_id, variant_id)

        # Default mode to random for config-id
        if args.mode is None:
            args.mode = "random"

        # Trained mode is not allowed with config-id
        if args.mode == "trained":
            raise ValueError("--mode trained requires --run-id, not --config-id")

        run = None
    else:
        # Run-id mode: load trained policy from checkpoint
        # Resolve run ID (handle @last symlink)
        run_id = args.run_id
        if run_id == "@last":
            from utils.run import LAST_RUN_DIR
            if not LAST_RUN_DIR.exists():
                raise FileNotFoundError("No @last run found. Train a model first.")
            run_id = LAST_RUN_DIR.resolve().name

        # Check if run exists locally, if not try to download from W&B
        run_dir = Run._resolve_run_dir(run_id)
        if not run_dir.exists():
            print(f"Run {run_id} not found locally. Attempting to download from W&B...")
            from utils.wandb_artifacts import download_run_artifact
            download_run_artifact(run_id)

        # Load run and config
        run = Run.load(run_id)
        config = run.load_config()

        # Default mode to trained for run-id
        if args.mode is None:
            args.mode = "trained"

        # For trained mode, require checkpoint
        if args.mode == "trained":
            assert run.best_checkpoint_dir is not None, "run has no best checkpoint"

    # Resolve seed argument
    if args.seed is None:
        # Default to test seed
        seed = config.seed_test
    elif args.seed in ["train", "val", "test"]:
        # Map stage names to corresponding seeds
        seeds = {
            "train": config.seed_train,
            "val": config.seed_val,
            "test": config.seed_test,
        }
        seed = seeds[args.seed]
    else:
        # Parse as integer
        seed = int(args.seed)

    # Seed all RNGs (Python, NumPy, PyTorch) for reproducibility
    set_random_seed(seed)

    # Build a single-env environment with human rendering
    # Force vectorization_mode='sync' to ensure render() is supported (ALE atari vectorization doesn't support it)
    env_overrides = {
        'n_envs': 1,
        'vectorization_mode': 'sync',
        'render_mode': "human" if not args.headless else None,
        'seed': seed
    }
    env = build_env_from_config(config, **env_overrides)

    # Attach a live observation bar printer for interactive play (vector-level wrapper)
    from gym_wrappers.vec_obs_printer import VecObsBarPrinter
    env = VecObsBarPrinter(env, bar_width=40, env_index=0, enable=True, target_episodes=target_episodes)

    # Handle different modes
    if args.mode in ["random", "user"]:
        # Manual control modes don't need policy
        play_episodes_manual(env, target_episodes, args.mode, args.step_by_step, args.fps)
        print("Done.")
        return

    # Trained mode: load policy
    assert run is not None, "run must be loaded for trained mode"
    # TODO: we should be loading the agent and having it run the episode
    policy_model, _ = load_policy_model_from_checkpoint(run.best_checkpoint_path, env, config)

    # Initialize rollout collector; step-by-step mode or FPS limiting uses single-step rollouts
    collector = RolloutCollector(
        env=env,
        policy_model=policy_model,
        n_steps=1 if (args.step_by_step or args.fps) else config.n_steps,
        **config.rollout_collector_hyperparams(),
    )

    if args.step_by_step:
        print(f"Playing {target_episodes} episode(s) step-by-step with render_mode='human'...")
        print("Press Enter to step, 'q' then Enter to quit.")
    else:
        print(f"Playing {target_episodes} episode(s) with render_mode='human'...")

    # Collect episodes until target episodes reached
    reported_episodes = 0
    frame_delay = 1.0 / args.fps if args.fps else 0
    while reported_episodes < target_episodes:
        if args.step_by_step:
            try:
                user = input("")
            except EOFError:
                user = ""
            if isinstance(user, str) and user.strip().lower() in {"q", "quit", "exit"}:
                break

        step_start = time.perf_counter()
        _ = collector.collect(deterministic=args.deterministic)

        # Apply FPS limiting if specified
        if frame_delay > 0:
            elapsed = time.perf_counter() - step_start
            sleep_time = max(0, frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
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
