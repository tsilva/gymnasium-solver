#!/usr/bin/env python3
"""
Play any NES ROM with random actions (biased for platformers).

Usage:
    python play_mario_random.py --list-roms                    # List available imported ROMs
    python play_mario_random.py --game SuperMarioBros-Nes      # Play specific game
    python play_mario_random.py --game Airstriker-Genesis --headless
    python play_mario_random.py --game SuperMarioBros-Nes --state Level1-1
"""
import retro
import numpy as np
import os
import platform
import argparse

def is_wsl():
    """Check if running in WSL."""
    return 'microsoft' in platform.uname().release.lower()

def is_headless():
    """Check if running in a headless environment."""
    return not os.environ.get('DISPLAY')

def list_available_roms():
    """List all available imported ROMs."""
    print("Available imported ROMs:\n")
    games = sorted(retro.data.list_games())

    if not games:
        print("No ROMs imported. Use `python -m retro.import /path/to/roms` to import ROMs.")
        return

    for game in games:
        print(f"  - {game}")

    print(f"\nTotal: {len(games)} game(s)")
    print("\nTo play a game, use: python play_mario_random.py --game <game_name>")

def main():
    parser = argparse.ArgumentParser(description='Play any NES ROM with random actions')
    parser.add_argument('--list-roms', action='store_true', help='List all available imported ROMs')
    parser.add_argument('--game', type=str, default='SuperMarioBros-Nes', help='Game name (e.g., SuperMarioBros-Nes)')
    parser.add_argument('--state', type=str, default=None, help='Starting state (e.g., Level1-1)')
    parser.add_argument('--headless', action='store_true', help='Force headless mode (no rendering window)')
    parser.add_argument('--render', action='store_true', help='Force human rendering mode')

    args = parser.parse_args()

    # Handle --list-roms
    if args.list_roms:
        list_available_roms()
        return

    # Determine render mode
    if args.headless:
        render_mode = 'rgb_array'
        print("Forced headless mode (no rendering window)\n")
    elif args.render:
        render_mode = 'human'
        print("Forced human rendering mode\n")
    else:
        # Auto-detect
        headless = is_headless()
        render_mode = 'rgb_array' if headless else 'human'
        if headless:
            print("Auto-detected headless environment (no DISPLAY)")
            print("Running without rendering window")
            print("Use --render to force human rendering if you have X11/WSLg configured\n")
        else:
            print("Auto-detected display available, using human rendering\n")

    # Create environment
    env_kwargs = {'game': args.game, 'render_mode': render_mode}
    if args.state:
        env_kwargs['state'] = args.state
    else:
        # Use default starting state for SuperMarioBros
        if args.game == 'SuperMarioBros-Nes':
            env_kwargs['state'] = 'Level1-1'

    env = retro.make(**env_kwargs)

    # Action space: MultiBinary(9)
    # Buttons: [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
    # Indices:  0    1      2       3     4    5     6      7     8

    RIGHT_BUTTON = 7
    A_BUTTON = 8  # Jump
    B_BUTTON = 0  # Run/Fire

    print(f"Starting {args.game} with right-biased random play...")
    if args.state:
        print(f"Starting state: {args.state}")
    print("Press Ctrl+C to stop")
    print("\nControls are biased (optimized for platformers):")
    print("  - RIGHT: 70% chance")
    print("  - JUMP (A): 30% chance")
    print("  - RUN (B): 40% chance")
    print("  - Other buttons: 10% chance each\n")

    episodes = 0

    try:
        while True:
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0

            episodes += 1
            print(f"\n=== Episode {episodes} ===")

            while not done:
                # Create random action with bias towards right
                action = np.zeros(9, dtype=np.int8)

                # 70% chance to press RIGHT
                if np.random.random() < 0.7:
                    action[RIGHT_BUTTON] = 1

                # 30% chance to press A (jump)
                if np.random.random() < 0.3:
                    action[A_BUTTON] = 1

                # 40% chance to press B (run)
                if np.random.random() < 0.4:
                    action[B_BUTTON] = 1

                # Small chance for other random buttons (10% each)
                for i in [4, 5, 6]:  # UP, DOWN, LEFT
                    if np.random.random() < 0.1:
                        action[i] = 1

                # Execute action
                # Gymnasium API returns 5 values: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Only call render() if not using rgb_array mode (it's automatic)
                if render_mode == 'human':
                    env.render()

                total_reward += reward
                steps += 1

                # Print progress every 100 steps
                if steps % 100 == 0:
                    print(f"Steps: {steps}, Total Reward: {total_reward:.0f}")

            print(f"Episode {episodes} finished!")
            print(f"  Total steps: {steps}")
            print(f"  Total reward: {total_reward:.0f}")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    finally:
        env.close()
        print(f"Played {episodes} episode(s)")

if __name__ == "__main__":
    main()
