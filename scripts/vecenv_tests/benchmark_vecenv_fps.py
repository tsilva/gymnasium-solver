"""Unified ALE throughput benchmark across three backends.

Backends:
- `alepy`  : ale-py native `AtariVectorEnv` (in-process, threaded)
- `gym`    : Gymnasium `SyncVectorEnv` (default) or `AsyncVectorEnv` with `--subproc`
- `sb3`    : Stable-Baselines3 `DummyVecEnv` (default) or `SubprocVecEnv` with `--subproc`

If `--backend` is omitted, runs all three backends (using the provided flags
like `--num-envs` and `--subproc`) and prints a ranked summary.

Examples:
- Single backend (Gym Async):
    python scripts/vecenv_tests/benchmark_vecenv_fps.py \
        --rom ALE/Pong-v5 --backend gym --subproc --frames 100000 --num-envs 8

- Compare all backends (Gym + SB3 respect --subproc flag):
    python scripts/vecenv_tests/benchmark_vecenv_fps.py \
        --rom ALE/Pong-v5 --frames 200000 --num-envs 8 --subproc
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict

import numpy as np

# Reuse shared helpers from the existing scripts to avoid duplication.
from benchmark_alepy_fps import (
    BenchmarkResult,
    ResolvedRom,
    detect_alepy_binary,
    resolve_rom,
    benchmark_vector_env as benchmark_alepy_vector_env,
)

# Import gym/sb3 benchmark entrypoints if available; raise clear guidance otherwise.
try:
    from benchmark_gym_vecenv_fps import benchmark_gym_sync_vec_env as benchmark_gym_vec
except ImportError as e:  # pragma: no cover - clarity for users missing gym
    benchmark_gym_vec = None  # type: ignore[assignment]
    _gym_import_error = e
else:
    _gym_import_error = None

try:
    from benchmark_sb3_vecenv_fps import benchmark_sb3_vec_env as benchmark_sb3_vec
except ImportError as e:  # pragma: no cover - clarity for users missing sb3
    benchmark_sb3_vec = None  # type: ignore[assignment]
    _sb3_import_error = e
else:
    _sb3_import_error = None


def _snake_to_camel(name: str) -> str:
    parts = [p for p in name.split("_") if p]
    return "".join(part.capitalize() for part in parts)


def infer_env_id(user_spec: str, rom: ResolvedRom) -> str:
    spec = user_spec.strip()
    if "/" in spec:
        return spec
    return f"ALE/{_snake_to_camel(rom.name)}-v5"


@dataclass
class RunOutcome:
    backend: str
    result: BenchmarkResult
    detail: str  # e.g., backend variant/config details


def _format_summary(outcome: RunOutcome, num_envs: int) -> str:
    per_env_fps = outcome.result.fps / max(1, num_envs)
    if outcome.result.frames == outcome.result.requested:
        frames_clause = f"{outcome.result.frames:,} frames"
    else:
        frames_clause = (
            f"{outcome.result.frames:,} frames executed ({outcome.result.requested:,} requested)"
        )
    return (
        f"Throughput: {outcome.result.fps:,.0f} FPS total | {per_env_fps:,.0f} FPS/env "
        f"over {frames_clause} in {outcome.result.elapsed:.2f}s "
        f"({outcome.detail})"
    )


def _run_alepy(rom: ResolvedRom, frames: int, num_envs: int, num_threads: Optional[int]) -> RunOutcome:
    result = benchmark_alepy_vector_env(
        rom,
        frames=frames,
        num_envs=num_envs,
        num_threads=num_threads,
    )
    return RunOutcome(
        backend="alepy",
        result=result,
        detail=f"backend=AtariVectorEnv, num_envs={num_envs}, num_threads={num_threads or num_envs}",
    )


def _run_gym(env_id: str, rom: ResolvedRom, frames: int, num_envs: int, use_subproc: bool) -> RunOutcome:
    if benchmark_gym_vec is None:
        raise ImportError(
            "Gym benchmark not available: failed to import benchmark_gym_vecenv_fps.py."
        ) from _gym_import_error
    result = benchmark_gym_vec(
        env_id, rom, frames=frames, num_envs=num_envs, use_subproc=use_subproc
    )
    backend = "AsyncVectorEnv" if use_subproc else "SyncVectorEnv"
    return RunOutcome(
        backend="gym",
        result=result,
        detail=f"backend={backend}, num_envs={num_envs}",
    )


def _run_sb3(env_id: str, rom: ResolvedRom, frames: int, num_envs: int, use_subproc: bool) -> RunOutcome:
    if benchmark_sb3_vec is None:
        raise ImportError(
            "SB3 benchmark not available: failed to import benchmark_sb3_vecenv_fps.py."
        ) from _sb3_import_error
    result = benchmark_sb3_vec(
        env_id, rom, frames=frames, num_envs=num_envs, use_subproc=use_subproc
    )
    backend = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
    return RunOutcome(
        backend="sb3",
        result=result,
        detail=f"backend={backend}, num_envs={num_envs}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified ALE throughput benchmark across ale-py, Gymnasium, and Stable-Baselines3 backends."
        )
    )
    parser.add_argument(
        "--rom",
        required=True,
        help="Path to ROM file or ALE game id (e.g. ALE/Pong-v5 or pong)",
    )
    parser.add_argument("--frames", type=int, default=200_000, help="Number of frames to benchmark")
    parser.add_argument(
        "--num-envs",
        "--nenvs",
        dest="num_envs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of vectorized environments (default: CPU cores)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Threads for alepy AtariVectorEnv (default: match --num-envs). Ignored by other backends.",
    )
    parser.add_argument(
        "--subproc",
        action="store_true",
        help="For gym/sb3: use process-based vector envs (AsyncVectorEnv/SubprocVecEnv)",
    )
    parser.add_argument(
        "--backend",
        choices=("alepy", "gym", "sb3"),
        default=None,
        help=(
            "Backend to run. If omitted, runs all (alepy + gym + sb3) sequentially "
            "and prints a ranked summary."
        ),
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

    # Print ale-py binary/arch info once for context (also used by gym/sb3 ALE envs).
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

    outcomes: list[RunOutcome] = []

    if args.backend is None:
        # Run all backends with provided flags (subproc applies to gym/sb3 only).
        outcomes.append(_run_alepy(rom, args.frames, args.num_envs, args.num_threads))
        outcomes.append(_run_gym(env_id, rom, args.frames, args.num_envs, args.subproc))
        outcomes.append(_run_sb3(env_id, rom, args.frames, args.num_envs, args.subproc))

        # Print individual summaries
        print()
        for oc in outcomes:
            print(_format_summary(oc, args.num_envs))

        # Rank and print winner
        fastest = max(outcomes, key=lambda oc: oc.result.fps)
        print(
            "\nFastest backend: "
            f"{fastest.backend} @ {fastest.result.fps:,.0f} FPS "
            f"({fastest.detail})"
        )
        return

    # Single-backend execution
    if args.backend == "alepy":
        outcome = _run_alepy(rom, args.frames, args.num_envs, args.num_threads)
    elif args.backend == "gym":
        outcome = _run_gym(env_id, rom, args.frames, args.num_envs, args.subproc)
    elif args.backend == "sb3":
        outcome = _run_sb3(env_id, rom, args.frames, args.num_envs, args.subproc)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unknown backend: {args.backend}")

    print()
    print(_format_summary(outcome, args.num_envs))


if __name__ == "__main__":
    main()

