"""Optimize ALE/Pong-v5 FPS using atari vectorization with systematic parameter search."""
import argparse
import itertools
import math
import time
from dataclasses import dataclass

import ale_py
import gymnasium as gym
import numpy as np
from tqdm import tqdm

gym.register_envs(ale_py)


@dataclass
class BenchmarkResult:
    n_envs: int
    batch_size: int | None
    thread_affinity_offset: int
    fps: float
    fps_per_env: float
    warmup_time: float
    bench_time: float

    def __str__(self):
        batch_str = str(self.batch_size) if self.batch_size is not None else "None"
        return (f"n_envs={self.n_envs:2d} batch_size={batch_str:4s} thread_offset={self.thread_affinity_offset:2d} | "
                f"{self.fps:>9,.0f} FPS | {self.fps_per_env:>7,.0f} FPS/env | "
                f"warmup={self.warmup_time:.2f}s bench={self.bench_time:.2f}s")


def benchmark_ale_atari(
    env_id: str,
    frames: int,
    n_envs: int,
    batch_size: int | None = None,
    thread_affinity_offset: int = 0,
    warmup_frames: int = 2048,
) -> BenchmarkResult:
    """Benchmark ALE atari vectorization with specified parameters."""
    kwargs = {
        "num_envs": n_envs,
        "vectorization_mode": None,  # Use ALE atari
        "repeat_action_probability": 0.0,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if thread_affinity_offset != 0:
        kwargs["thread_affinity_offset"] = thread_affinity_offset

    vec_env = gym.make_vec(env_id, **kwargs)

    try:
        obs, _ = vec_env.reset()
        assert obs.shape == (n_envs, 4, 84, 84), f"Shape mismatch: {obs.shape}"

        actions = np.zeros(n_envs, dtype=np.int64)

        # Warmup
        warmup_steps = max(1, math.ceil(warmup_frames / n_envs))
        warmup_start = time.perf_counter()
        for _ in range(warmup_steps):
            vec_env.step(actions)
        warmup_time = time.perf_counter() - warmup_start

        # Benchmark
        bench_steps = max(1, math.ceil(frames / n_envs))
        bench_start = time.perf_counter()
        for _ in range(bench_steps):
            vec_env.step(actions)
        bench_time = time.perf_counter() - bench_start

        fps = bench_steps * n_envs / bench_time
        fps_per_env = fps / n_envs

        return BenchmarkResult(
            n_envs=n_envs,
            batch_size=batch_size,
            thread_affinity_offset=thread_affinity_offset,
            fps=fps,
            fps_per_env=fps_per_env,
            warmup_time=warmup_time,
            bench_time=bench_time,
        )
    finally:
        vec_env.close()


def grid_search(
    env_id: str,
    frames: int,
    n_envs_list: list[int],
    batch_sizes: list[int | None],
    thread_offsets: list[int],
    warmup_frames: int = 2048,
) -> list[BenchmarkResult]:
    """Run grid search over all parameter combinations."""
    results = []
    total_configs = len(n_envs_list) * len(batch_sizes) * len(thread_offsets)

    print(f"Running grid search: {total_configs} configurations")
    print(f"n_envs: {n_envs_list}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"thread_offsets: {thread_offsets}")
    print(f"frames per test: {frames:,}")
    print()

    best_result = None

    for n_envs, batch_size, thread_offset in tqdm(
        itertools.product(n_envs_list, batch_sizes, thread_offsets),
        total=total_configs,
        desc="Benchmarking",
        unit="config"
    ):
        try:
            result = benchmark_ale_atari(
                env_id=env_id,
                frames=frames,
                n_envs=n_envs,
                batch_size=batch_size,
                thread_affinity_offset=thread_offset,
                warmup_frames=warmup_frames,
            )
            results.append(result)

            # Update best result
            if best_result is None or result.fps > best_result.fps:
                best_result = result

            # Print this trial's result
            tqdm.write(f"\n{'='*80}")
            tqdm.write(f"Trial {len(results)}/{total_configs} Result:")
            tqdm.write(f"  {result}")
            tqdm.write(f"\nBest Config So Far:")
            tqdm.write(f"  {best_result}")
            tqdm.write(f"  (n_envs={best_result.n_envs}, batch_size={best_result.batch_size}, thread_offset={best_result.thread_affinity_offset})")
            tqdm.write(f"{'='*80}\n")

        except Exception as e:
            tqdm.write(f"FAILED: n_envs={n_envs} batch_size={batch_size} thread_offset={thread_offset} | {e}")

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print summary of benchmark results."""
    if not results:
        print("\nNo successful results")
        return

    results_sorted = sorted(results, key=lambda r: r.fps, reverse=True)

    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by total FPS)")
    print("="*80)
    for i, r in enumerate(results_sorted[:10], 1):
        print(f"{i:2d}. {r}")

    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS (by FPS/env)")
    print("="*80)
    results_by_fps_per_env = sorted(results, key=lambda r: r.fps_per_env, reverse=True)
    for i, r in enumerate(results_by_fps_per_env[:10], 1):
        print(f"{i:2d}. {r}")

    best = results_sorted[0]
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION")
    print("="*80)
    print(f"n_envs: {best.n_envs}")
    print(f"batch_size: {best.batch_size if best.batch_size is not None else 'None (default)'}")
    print(f"thread_affinity_offset: {best.thread_affinity_offset}")
    print(f"Total FPS: {best.fps:,.0f}")
    print(f"FPS per env: {best.fps_per_env:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize ALE/Pong-v5 FPS with atari vectorization")
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--frames", type=int, default=100_000, help="Frames per benchmark run")
    parser.add_argument("--warmup-frames", type=int, default=2048, help="Frames for warmup")

    # Grid search parameters
    parser.add_argument("--n-envs", type=int, nargs="+",
                        help="List of n_envs to test (default: 1,2,4,6,8,12,16,24,32)")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        help="List of batch_sizes to test (default: None,1,2,4,8)")
    parser.add_argument("--thread-offsets", type=int, nargs="+",
                        help="List of thread_affinity_offsets to test (default: 0,1,2)")

    args = parser.parse_args()

    # Default parameter ranges
    if args.n_envs is None:
        args.n_envs = [8, 16, 32]
    
    if args.batch_sizes is None:
        args.batch_sizes = [None, 1, 2, 4, 8]

    if args.thread_offsets is None:
        args.thread_offsets = [0, 1, 2]
       
    # Convert None string to actual None
    batch_sizes = [None if b == "None" else b for b in args.batch_sizes]

    results = grid_search(
        env_id=args.env_id,
        frames=args.frames,
        n_envs_list=args.n_envs,
        batch_sizes=batch_sizes,
        thread_offsets=args.thread_offsets,
        warmup_frames=args.warmup_frames,
    )

    print_summary(results)
