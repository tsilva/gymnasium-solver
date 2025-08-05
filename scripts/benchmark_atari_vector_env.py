from ale_py.vector_env import AtariVectorEnv
from sklearn.conftest import environ




import gymnasium as gym
import time

import ale_py
import gymnasium
gymnasium.register_envs(ale_py)
def benchmark(env_id="ALE/Pong-v5", num_steps=10000):
    env = AtariVectorEnv(
        game="ms_pacman",        # ROM id (camel case)
        num_envs=8,
        frameskip=4,
        grayscale=True,
        stack_num=4,
        img_height=84,
        img_width=84,
        noop_max=30,
        use_fire_reset=False,
        episodic_life=False,
        repeat_action_probability=0.0,  # sticky actions
        obs_type="ram",  # use RAM observations
        #render_mode=None
    )


    obs, info = env.reset()

    start = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated.any() or truncated.any():
            obs, info = env.reset()
    elapsed = time.time() - start
    
    fps = num_steps / elapsed
    env.close()
    return fps

if __name__ == "__main__":
    print("Benchmarking Pong-v5 environment...")
    fps = benchmark()
    print(f"FPS: {fps:.2f}")