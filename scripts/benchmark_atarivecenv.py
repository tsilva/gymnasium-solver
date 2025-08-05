import os
import time
import numpy as np
from ale_py.vector_env import AtariVectorEnv

# --- Throughput-oriented defaults for an M1 MBP ---
NUM_ENVS = os.cpu_count() or 8          # start with one env per core
NUM_THREADS = NUM_ENVS                  # match threads to envs
TOTAL_STEPS = 200_000                   # total single-env steps to run for timing

envs = AtariVectorEnv(
    game="pong",                        # ALE ROM id
    num_envs=NUM_ENVS,
    # Disable preprocessing / keep raw frames
    grayscale=False,                    # RGB
    stack_num=1,                        # no frame stacking
    # img_height=None, img_width=None   # keep native 210x160
    # Throughput-related knobs
    num_threads=NUM_THREADS,
    thread_affinity_offset=0,           # let OS schedule threads
    noop_max=0,                         # no initial no-ops
    # If your version supports these, uncomment to squeeze a bit more:
    # frameskip=1,                      # minimize per-step emulator work
    # full_action_space=False,          # minimal action set
    # repeat_action_probability=0.0,    # deterministic (slightly faster)
    # render_mode=None,                 # make sure nothing is drawn
)

# Reset once
obs, info = envs.reset()

steps_done = 0
start = time.perf_counter()

# Step in a tight loop until TOTAL_STEPS reached across all envs
while steps_done < TOTAL_STEPS:
    obs, rewards, terminations, truncations, infos = envs.step(envs.action_space.sample())
    steps_done += envs.num_envs

    # Cheap partial resets when available (avoid resetting all envs)
    dones = np.logical_or(terminations, truncations)
    if np.any(dones):
        # Try common APIs without branching cost elsewhere
        try:
            # Some vector envs accept indices
            envs.reset(np.flatnonzero(dones))
        except TypeError:
            try:
                # Some expose a boolean-mask reset
                envs.reset_done(dones)
            except AttributeError:
                # Some auto-reset; if not, you can ignore for pure throughput timing
                pass

elapsed = time.perf_counter() - start
fps = steps_done / elapsed
print(f"FPS: {fps:.2f}  (steps={steps_done}, elapsed={elapsed:.2f}s, num_envs={envs.num_envs}, threads={NUM_THREADS})")

envs.close()