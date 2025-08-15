# benchmark_sb3_vecenv_make.py
import os
import time

import ale_py
import gymnasium  # needed for register_envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

gymnasium.register_envs(ale_py)

# ---------- Config ----------
ENV_ID = "ALE/Pong-v5"                 # ALE ROM id (Gymnasium)
TOTAL_STEPS = 200_000                  # total single-env steps to run for timing
NUM_ENVS = os.cpu_count() or 8         # start with one env per CPU core
USE_SUBPROC = True                     # try True/False to see which is faster on your M1

# Optional: reduce thread contention from BLAS/OpenMP libs
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def main():
    vec_env_cls = SubprocVecEnv if USE_SUBPROC else DummyVecEnv
    vec_env_kwargs = {"start_method": "spawn"} if USE_SUBPROC else {}

    # Build the vectorized env with SB3 helper
    vec_env = make_vec_env(
        ENV_ID,
        n_envs=NUM_ENVS,
        env_kwargs={"render_mode": None, "obs_type": "ram"},
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    # Reset once
    obs = vec_env.reset()

    steps_done = 0
    start = time.perf_counter()

    # Tight loop; SB3 VecEnvs auto-reset finished envs internally
    while steps_done < TOTAL_STEPS:
        actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        steps_done += vec_env.num_envs

    elapsed = time.perf_counter() - start
    fps = steps_done / elapsed
    print(
        f"FPS: {fps:.2f}  (steps={steps_done}, elapsed={elapsed:.2f}s, "
        f"num_envs={vec_env.num_envs}, subproc={USE_SUBPROC})"
    )

    vec_env.close()


if __name__ == "__main__":
    main()