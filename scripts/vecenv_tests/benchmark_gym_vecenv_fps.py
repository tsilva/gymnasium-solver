"""Benchmark ALE throughput via Gymnasium's Sync/Async VectorEnv.

Matches the CLI of `benchmark_alepy_fps.py` and `benchmark_sb3_vecenv_fps.py`.
This uses `gym.vector.SyncVectorEnv` by default and switches to
`gym.vector.AsyncVectorEnv` when `--subproc` is specified. Measures
raw stepping FPS with a fixed action.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from benchmark_alepy_fps import (
    BenchmarkResult,
    ResolvedRom,
    detect_alepy_binary,
    resolve_rom,
)


def _snake_to_camel(name: str) -> str:
    parts = [p for p in name.split("_") if p]
    return "".join(part.capitalize() for part in parts)


def infer_env_id(user_spec: str, rom: ResolvedRom) -> str:
    spec = user_spec.strip()
    if "/" in spec:
        return spec
    return f"ALE/{_snake_to_camel(rom.name)}-v5"


def benchmark_gym_sync_vec_env(
    env_id: str,
    rom: ResolvedRom,
    frames: int,
    num_envs: int,
    *,
    use_subproc: bool,
    warmup_steps: int = 256,
    chunk_steps: int = 256,
) -> BenchmarkResult:
    if rom.is_custom:
        raise ValueError(
            "Custom ROM paths are not supported by Gymnasium ALE envs; "
            "pass a packaged ROM id like 'ALE/Pong-v5'."
        )
    assert frames > 0, "Frame count must be positive"
    assert num_envs > 0, "Number of environments must be positive"
    assert chunk_steps > 0, "Chunk size must be positive"

    env_kwargs = dict(
        # frameskip=1,
        # repeat_action_probability=0.0,
        # full_action_space=False,
        # noop_max=0,
        # render_mode=None,
        # obs_type="rgb",
    )

    def make_env():
        return gym.make(env_id, **env_kwargs)

    EnvClass = AsyncVectorEnv if use_subproc else SyncVectorEnv
    vec_env = EnvClass([make_env for _ in range(num_envs)])

    try:
        obs, _infos = vec_env.reset()
        _ = obs  # silence unused warning

        # For Gymnasium VectorEnvs, the single action space is the base env's space
        if isinstance(vec_env.single_action_space, Discrete):
            action_value = 0
        else:
            raise ValueError("This benchmark only supports discrete ALE actions.")
        actions = np.full((num_envs,), action_value, dtype=np.int64)

        warmup_iters = max(1, math.ceil(warmup_steps / max(1, num_envs)))
        for _ in range(warmup_iters):
            vec_env.step(actions)

        frames_done = 0
        steps_remaining = max(1, math.ceil(frames / num_envs))
        start = time.perf_counter()

        while steps_remaining > 0:
            steps_this_chunk = min(chunk_steps, steps_remaining)
            for _ in range(steps_this_chunk):
                vec_env.step(actions)
                frames_done += num_envs
            steps_remaining -= steps_this_chunk

        elapsed = time.perf_counter() - start
    finally:
        vec_env.close()

    return BenchmarkResult(frames=frames_done, elapsed=elapsed, requested=frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ALE FPS via Gymnasium Sync/Async VectorEnv."
    )
    parser.add_argument("--rom", required=True, help="Path to ROM file or ALE game id (e.g. ALE/Pong-v5)")
    parser.add_argument("--frames", type=int, default=200_000, help="Number of frames to benchmark")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of vectorized environments (default: CPU cores)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Unused placeholder for parity with ALE vector benchmark.",
    )
    parser.add_argument(
        "--subproc",
        action="store_true",
        help="Use AsyncVectorEnv instead of SyncVectorEnv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_threads is not None and args.num_threads <= 0:
        raise ValueError("--num-threads must be positive when provided")

    rom = resolve_rom(args.rom)
    env_id = infer_env_id(args.rom, rom)

    binary_info = detect_alepy_binary()
    print("Python architecture:", binary_info.python_arch)
    if binary_info.library_path:
        print("ALE shared library:", binary_info.library_path)
    if binary_info.file_description:
        print("Binary description:", binary_info.file_description)
    print(binary_info.arch_note)
    print("ROM path:", rom.path)
    print("ROM id:", rom.name)
    print("Env id:", env_id)
    backend = "AsyncVectorEnv" if args.subproc else "SyncVectorEnv"

    result = benchmark_gym_sync_vec_env(
        env_id,
        rom,
        frames=args.frames,
        num_envs=args.num_envs,
        use_subproc=args.subproc,
    )
    per_env_fps = result.fps / args.num_envs
    if result.frames == result.requested:
        frames_clause = f"{result.frames:,} frames"
    else:
        frames_clause = (
            f"{result.frames:,} frames executed ({result.requested:,} requested)"
        )
    print(
        "\nThroughput: "
        f"{result.fps:,.0f} FPS total | {per_env_fps:,.0f} FPS/env "
        f"over {frames_clause} in {result.elapsed:.2f}s "
        f"(backend={backend}, num_envs={args.num_envs})"
    )


if __name__ == "__main__":
    main()
