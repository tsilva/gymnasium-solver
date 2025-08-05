import gymnasium as gym
import time

import ale_py
import gymnasium
gymnasium.register_envs(ale_py)
def benchmark(env_id="ALE/Pong-v5", num_steps=10000):
    env = gym.make(env_id, render_mode=None)  # disable rendering for max speed
    obs, info = env.reset()
    
    start = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    elapsed = time.time() - start
    
    fps = num_steps / elapsed
    env.close()
    return fps

if __name__ == "__main__":
    print("Benchmarking Pong-v5 environment...")
    fps = benchmark()
    print(f"FPS: {fps:.2f}")