"""Measure raw ALE emulator throughput and report the compiled binary architecture.

Usage example:
    python scripts/benchmark_alepy_fps.py --rom ALE/Pong-v5 --frames 200000 --num-envs 8

The script keeps Python overhead minimal by:
- vectorizing over ALE sub-envs with a pre-bound action batch
- using a single fixed action to avoid sampling overhead
- chunking work inside the timing loop with minimal Python bookkeeping
- resetting only when the emulator signals game over

It also reports whether the ale-py shared library is ARM64 (Apple Silicon)
or x86_64 (Rosetta on Apple Silicon).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import ale_py
import numpy as np
from ale_py.vector_env import AtariVectorEnv


@dataclass
class AleBinaryInfo:
    python_arch: str
    library_path: Optional[str]
    file_description: Optional[str]
    arch_note: str


@dataclass
class ResolvedRom:
    name: str
    path: str
    is_custom: bool


def detect_alepy_binary() -> AleBinaryInfo:
    """Inspect the ale-py shared library and classify the host/target architecture."""
    lib_dir = os.path.dirname(ale_py.__file__)
    so_candidates = glob.glob(os.path.join(lib_dir, "*.so"))

    if not so_candidates:
        return AleBinaryInfo(
            python_arch=platform.machine(),
            library_path=None,
            file_description=None,
            arch_note="No ale-py shared library found; check installation.",
        )

    so_path = so_candidates[0]
    file_description: Optional[str] = None
    try:
        result = subprocess.run(
            ["file", so_path],
            check=False,
            capture_output=True,
            text=True,
        )
        file_description = result.stdout.strip() or None
    except FileNotFoundError:
        file_description = "`file` command not available"

    arch_note = "Unknown architecture"
    if file_description:
        text = file_description.lower()
        if "arm64" in text or "aarch64" in text:
            arch_note = "✅ ALE build is arm64 (Apple Silicon native)."
        elif "x86_64" in text:
            arch_note = "⚠️ ALE build is x86_64 (likely running under Rosetta)."

    return AleBinaryInfo(
        python_arch=platform.machine(),
        library_path=so_path,
        file_description=file_description,
        arch_note=arch_note,
    )


_VERSION_PATTERN = re.compile(r"-v\d+$", re.IGNORECASE)
_OBSERVATION_SUFFIXES = ("-ram", "-rgb")
_MODE_SUFFIXES = ("NoFrameskip", "Deterministic")


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case without allocating intermediate lists."""
    name = name.replace("-", "_")
    name = re.sub("(.)([A-Z][a-z0-9]+)", r"\\1_\\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\\1_\\2", name)
    return name.replace("__", "_").strip("_").lower()


def _candidate_rom_names(spec: str) -> Iterable[str]:
    trimmed = spec.strip()
    if not trimmed:
        return ()

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    normalized = trimmed
    if normalized.startswith("ALE/"):
        normalized = normalized[4:]

    normalized = normalized.strip("/_")
    normalized = _VERSION_PATTERN.sub("", normalized)

    lower_normalized = normalized.lower()
    for suffix in _OBSERVATION_SUFFIXES:
        if lower_normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            lower_normalized = normalized.lower()
            break

    for suffix in _MODE_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]

    normalized = normalized.strip("-_ ")
    if normalized:
        _add(normalized)
        _add(normalized.lower())
        snake = _camel_to_snake(normalized)
        _add(snake)
        _add(snake.replace("_", ""))
        capitalized = normalized.capitalize()
        _add(capitalized)

    _add(trimmed)
    _add(trimmed.lower())

    return tuple(candidates)


def resolve_rom(rom: str) -> ResolvedRom:
    """Resolve a ROM from file path, ALE env id (e.g. ALE/Pong-v5), or ROM name."""
    if os.path.isfile(rom):
        return ResolvedRom(
            name=os.path.splitext(os.path.basename(rom))[0],
            path=rom,
            is_custom=True,
        )

    from ale_py import roms

    last_error: Optional[Exception] = None
    for candidate in _candidate_rom_names(rom):
        try:
            resolved = roms.get_rom_path(candidate)
        except KeyError as exc:
            last_error = exc
            continue

        if not resolved:
            continue

        if os.path.isfile(resolved):
            return ResolvedRom(name=candidate, path=resolved, is_custom=False)

    raise FileNotFoundError(
        f"ROM '{rom}' not found. Provide a file path, an ALE env id (e.g. ALE/Pong-v5), or install ale-py ROMs."
    ) from last_error


@dataclass
class BenchmarkResult:
    frames: int
    elapsed: float
    requested: int

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else float("inf")


def benchmark_vector_env(
    rom: ResolvedRom,
    frames: int,
    num_envs: int,
    *,
    num_threads: Optional[int] = None,
    warmup_steps: int = 256,
    chunk_steps: int = 256,
) -> BenchmarkResult:
    """Run a vectorized ALE benchmark for a fixed number of frames."""
    assert frames > 0, "Frame count must be positive"
    assert num_envs > 0, "Number of environments must be positive"
    assert chunk_steps > 0, "Chunk size must be positive"

    if rom.is_custom:
        raise ValueError(
            "Custom ROM paths are not supported by the ALE vector interface; "
            "pass a packaged ROM id like 'pong'."
        )

    threads = num_threads if num_threads is not None else num_envs
    env = AtariVectorEnv(
        game=rom.name,
        num_envs=num_envs,
        num_threads=threads,
        grayscale=False,
        stack_num=1,
        frameskip=1,
        img_height=210,
        img_width=160,
        maxpool=False,
        noop_max=0,
        repeat_action_probability=0.0,
        reward_clipping=False,
        full_action_space=False,
        use_fire_reset=True,
    )

    try:
        reset = env.reset
        step = env.step
        reset()

        action_value = 0
        if hasattr(env, "single_action_space"):
            action_value = int(getattr(env.single_action_space, "start", 0))
        actions = np.full((num_envs,), action_value, dtype=np.int32)

        reset_mask = np.zeros((num_envs, 1), dtype=np.bool_)

        warmup_iters = max(1, math.ceil(warmup_steps / max(1, num_envs)))
        for _ in range(warmup_iters):
            _, _, terminations, truncations, _ = step(actions)
            dones = np.logical_or(terminations, truncations)
            if np.any(dones):
                reset_mask[:, 0] = dones
                reset(options={"reset_mask": reset_mask})
                reset_mask[:, 0] = False

        reset()

        frames_done = 0
        steps_remaining = max(1, math.ceil(frames / num_envs))
        start = time.perf_counter()

        while steps_remaining > 0:
            steps_this_chunk = min(chunk_steps, steps_remaining)
            for _ in range(steps_this_chunk):
                _, _, terminations, truncations, _ = step(actions)
                frames_done += num_envs
                dones = np.logical_or(terminations, truncations)
                if np.any(dones):
                    reset_mask[:, 0] = dones
                    reset(options={"reset_mask": reset_mask})
                    reset_mask[:, 0] = False
            steps_remaining -= steps_this_chunk

        elapsed = time.perf_counter() - start
    finally:
        env.close()

    return BenchmarkResult(frames=frames_done, elapsed=elapsed, requested=frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ALE emulator FPS and report build arch.")
    parser.add_argument("--rom", required=True, help="Path to ROM file or ALE game id (e.g. pong)")
    parser.add_argument("--frames", type=int, default=200_000, help="Number of frames to benchmark")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of ALE environments to vectorize (default: CPU cores)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Threads to dedicate to ALE (default: match --num-envs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_threads is not None and args.num_threads <= 0:
        raise ValueError("--num-threads must be positive when provided")

    rom = resolve_rom(args.rom)

    binary_info = detect_alepy_binary()
    print("Python architecture:", binary_info.python_arch)
    if binary_info.library_path:
        print("ALE shared library:", binary_info.library_path)
    if binary_info.file_description:
        print("Binary description:", binary_info.file_description)
    print(binary_info.arch_note)
    print("ROM path:", rom.path)
    print("ROM id:", rom.name)

    result = benchmark_vector_env(
        rom,
        frames=args.frames,
        num_envs=args.num_envs,
        num_threads=args.num_threads,
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
        f"(num_envs={args.num_envs}, num_threads={args.num_threads or args.num_envs})"
    )


if __name__ == "__main__":
    main()
