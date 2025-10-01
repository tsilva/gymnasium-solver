"""Benchmark ALE throughput using gym.make_vec with different vectorization backends.

Backends:
- `sync`  : SyncVectorEnv
- `async` : AsyncVectorEnv
- None    : Routes to AleVecEnv when available

If `--backend` is omitted, runs all backends and prints a ranked summary.

Examples:
    python scripts/benchmark_vecenv_fps.py --rom ALE/Pong-v5 --backend async --frames 100000 --num-envs 8
    python scripts/benchmark_vecenv_fps.py --rom ALE/Pong-v5 --frames 200000 --num-envs 8
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)


@dataclass
class BenchmarkResult:
    frames: int
    elapsed: float

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else float("inf")


def benchmark(env_id: str, frames: int, num_envs: int, vectorization_mode: Optional[str]) -> BenchmarkResult:
    import numpy as np
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

    if vectorization_mode is None:
        assert hasattr(gym, "make_vec"), "Gymnasium version does not expose gym.make_vec"
        vec_env = gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode=None,
            use_fire_reset=True,
            reward_clipping=True,
            repeat_action_probability=0.0,
        )
    else:
        def make_env():
            env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
            env = gym.wrappers.AtariPreprocessing(
                env, noop_max=30, frame_skip=4, screen_size=84,
                terminal_on_life_loss=False, grayscale_obs=True,
                grayscale_newaxis=False, scale_obs=False,
            )
            return gym.wrappers.FrameStackObservation(env, stack_size=4, padding_type="zero")

        VecEnvClass = AsyncVectorEnv if vectorization_mode == "async" else SyncVectorEnv
        vec_env = VecEnvClass([make_env for _ in range(num_envs)])

    try:
        obs, _ = vec_env.reset()
        assert obs.shape == (num_envs, 4, 84, 84), (
            f"Shape mismatch for mode={vectorization_mode}: expected {(num_envs, 4, 84, 84)}, got {obs.shape}"
        )

        actions = np.zeros(num_envs, dtype=np.int64)

        # Warmup
        for _ in range(max(1, math.ceil(256 / num_envs))):
            vec_env.step(actions)

        # Benchmark
        steps = max(1, math.ceil(frames / num_envs))
        start = time.perf_counter()
        for _ in range(steps):
            vec_env.step(actions)
        elapsed = time.perf_counter() - start

        return BenchmarkResult(frames=steps * num_envs, elapsed=elapsed)
    finally:
        vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ALE vectorization backends via gym.make_vec")
    parser.add_argument("--rom", required=True, help="ALE game id (e.g. ALE/Pong-v5)")
    parser.add_argument("--frames", type=int, default=200_000, help="Number of frames to benchmark")
    parser.add_argument("--num-envs", type=int, default=os.cpu_count() or 1, help="Number of parallel environments")
    parser.add_argument("--backend", choices=("sync", "async", "none"), help="Vectorization backend (omit to run all)")
    args = parser.parse_args()

    assert args.num_envs > 0, "--num-envs must be positive"
    env_id = args.rom

    backends = [None if args.backend == "none" else args.backend] if args.backend else ["sync", "async", None]
    results = []

    for mode in backends:
        try:
            result = benchmark(env_id, args.frames, args.num_envs, mode)
            mode_str = mode or "None (AleVecEnv)"
            results.append((mode_str, result))
            print(f"{mode_str:20s}: {result.fps:>8,.0f} FPS | {result.fps / args.num_envs:>6,.0f} FPS/env | {result.elapsed:.2f}s")
        except Exception as e:
            print(f"{mode or 'None':20s}: Failed ({e})")

    if len(results) > 1:
        fastest = max(results, key=lambda r: r[1].fps)
        print(f"\nFastest: {fastest[0]} @ {fastest[1].fps:,.0f} FPS")


if __name__ == "__main__":
    main()
