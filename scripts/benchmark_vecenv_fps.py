"""Benchmark ALE vectorization backends: sync (SyncVectorEnv), async (AsyncVectorEnv), none (AleVecEnv)."""
import argparse
import math
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

gym.register_envs(ale_py)


def benchmark(env_id: str, frames: int, num_envs: int, mode: str | None) -> float:
    if mode is None:
        vec_env = gym.make_vec(env_id, num_envs, vectorization_mode=None,
                               use_fire_reset=True, reward_clipping=True, repeat_action_probability=0.25)
    else:
        def make_env():
            env = gym.make(env_id, frameskip=1, repeat_action_probability=0.25)
            env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                                                   terminal_on_life_loss=False, grayscale_obs=True,
                                                   grayscale_newaxis=False, scale_obs=False)
            return gym.wrappers.FrameStackObservation(env, stack_size=4, padding_type="zero")
        vec_env = (AsyncVectorEnv if mode == "async" else SyncVectorEnv)([make_env for _ in range(num_envs)])

    try:
        obs, _ = vec_env.reset()
        assert obs.shape == (num_envs, 4, 84, 84), f"Shape mismatch: {obs.shape}"
        actions = np.zeros(num_envs, dtype=np.int64)
        for _ in range(max(1, math.ceil(256 / num_envs))):  # warmup
            vec_env.step(actions)
        steps = max(1, math.ceil(frames / num_envs))
        start = time.perf_counter()
        for _ in range(steps):
            vec_env.step(actions)
        return steps * num_envs / (time.perf_counter() - start)
    finally:
        vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ALE vectorization backends")
    parser.add_argument("--rom", required=True, help="ALE env id (e.g. ALE/Pong-v5)")
    parser.add_argument("--frames", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--backend", choices=("sync", "async", "none"))
    args = parser.parse_args()

    modes = [None if args.backend == "none" else args.backend] if args.backend else ["sync", "async", None]
    results = []
    for mode in modes:
        try:
            fps = benchmark(args.rom, args.frames, args.num_envs, mode)
            mode_str = mode or "none"
            results.append((mode_str, fps))
            print(f"{mode_str:5s}: {fps:>8,.0f} FPS | {fps / args.num_envs:>6,.0f} FPS/env")
        except Exception as e:
            print(f"{mode or 'none':5s}: Failed ({e})")

    if len(results) > 1:
        print(f"\nFastest: {max(results, key=lambda r: r[1])[0]}")
