import argparse
import time

import gymnasium as gym


# BipedalWalker-v3
# CarRacing-v3
# LunarLander-v3
# Blackjack-v1
# CliffWalking-v0
# FrozenLake-v1
# Ant-v5
# HalfCheetah-v5
# Hopper-v5
# Humanoid-v5
# HumanoidStandup-v5
# InvertedDoublePendulum-v5
# InvertedPendulum-v5
# Pusher-v5
# Reacher-v5
# Swimmer-v5
# Walker2d-v5
def main():
    parser = argparse.ArgumentParser(description="Run a Gymnasium environment with random actions.")
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name (default: CartPole-v1)"
    )
    args = parser.parse_args()

    env = None
    try:
        env = gym.make(args.env, render_mode="human")
        obs, info = env.reset()

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()

            time.sleep(0.02)  # ~50 FPS
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if env is not None:
            env.close()
        print("[INFO] Environment closed.")

if __name__ == "__main__":
    main()
