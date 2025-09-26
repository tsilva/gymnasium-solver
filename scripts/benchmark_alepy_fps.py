"""Measure raw ALE emulator throughput and report the compiled binary architecture.

Usage example:
    python scripts/benchmark_alepy_fps.py --rom ALE/Pong-v5 --frames 200000

The script keeps Python overhead minimal by:
- pre-binding the `act` callable
- using a single fixed action
- unrolling the inner loop
- resetting only when the emulator signals game over

It also reports whether the ale-py shared library is ARM64 (Apple Silicon)
or x86_64 (Rosetta on Apple Silicon).
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import ale_py


@dataclass
class AleBinaryInfo:
    python_arch: str
    library_path: Optional[str]
    file_description: Optional[str]
    arch_note: str


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


def resolve_rom_path(rom: str) -> str:
    """Resolve a ROM path from file path, ALE env id (e.g. ALE/Pong-v5), or ROM name."""
    if os.path.isfile(rom):
        return rom

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
            return resolved

    raise FileNotFoundError(
        f"ROM '{rom}' not found. Provide a file path, an ALE env id (e.g. ALE/Pong-v5), or install ale-py ROMs."
    ) from last_error


@dataclass
class BenchmarkResult:
    frames: int
    elapsed: float

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else float("inf")


def benchmark_ale(rom_path: str, frames: int, unroll: int = 8, chunk: int = 1024) -> BenchmarkResult:
    """Run the ALE emulator for a fixed number of frames with minimal Python overhead."""
    assert frames > 0, "Frame count must be positive"
    assert chunk % unroll == 0, "Chunk size must be divisible by unroll factor"

    ale = ale_py.ALEInterface()
    ale.setInt("frame_skip", 1)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setBool("display_screen", False)
    ale.setBool("sound", False)

    ale.loadROM(rom_path)
    minimal_actions = ale.getMinimalActionSet()
    action = minimal_actions[0]
    act = ale.act  # bind once to avoid attribute lookups
    reset_game = ale.reset_game
    game_over = ale.game_over

    warmup_steps = 512
    for _ in range(warmup_steps):
        act(action)
        if game_over():
            reset_game()

    frames_remaining = frames
    frames_done = 0
    start = time.perf_counter()

    while frames_remaining >= chunk:
        loops = chunk // unroll
        for _ in range(loops):
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
        frames_remaining -= chunk
        frames_done += chunk
        if game_over():
            reset_game()

    if frames_remaining:
        loops = frames_remaining // unroll
        for _ in range(loops):
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
            act(action)
        tail = frames_remaining % unroll
        for _ in range(tail):
            act(action)
        frames_done += frames_remaining
        if game_over():
            reset_game()

    elapsed = time.perf_counter() - start
    return BenchmarkResult(frames=frames_done, elapsed=elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ALE emulator FPS and report build arch.")
    parser.add_argument("--rom", required=True, help="Path to ROM file or ALE game id (e.g. pong)")
    parser.add_argument("--frames", type=int, default=200_000, help="Number of frames to benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rom_path = resolve_rom_path(args.rom)

    binary_info = detect_alepy_binary()
    print("Python architecture:", binary_info.python_arch)
    if binary_info.library_path:
        print("ALE shared library:", binary_info.library_path)
    if binary_info.file_description:
        print("Binary description:", binary_info.file_description)
    print(binary_info.arch_note)

    result = benchmark_ale(rom_path, frames=args.frames)
    print(
        f"\nThroughput: {result.fps:,.0f} FPS over {result.frames:,} frames in {result.elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
