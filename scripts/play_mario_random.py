#!/usr/bin/env python3
"""
Play Super Mario Bros with random actions biased towards moving right.
"""
import retro
import numpy as np

def main():
    # Create environment
    env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')

    # Action space: MultiBinary(9)
    # Buttons: [B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
    # Indices:  0    1      2       3     4    5     6      7     8

    RIGHT_BUTTON = 7
    A_BUTTON = 8  # Jump
    B_BUTTON = 0  # Run/Fire

    print("Starting Super Mario Bros with right-biased random play...")
    print("Press Ctrl+C to stop")
    print("\nControls are biased:")
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
