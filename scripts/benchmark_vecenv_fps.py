"""Unified ALE throughput benchmark across three backends.

Backends:
- `alepy`    : ale-py native `AtariVectorEnv` (in-process, threaded)
- `gym`      : Gymnasium `SyncVectorEnv` (default) or `AsyncVectorEnv` with `--subproc`
- `gym-make` : Gymnasium `make_vec(..., vectorization_mode=sync|async)`
- `sb3`      : Stable-Baselines3 `DummyVecEnv` (default) or `SubprocVecEnv` with `--subproc`

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
from typing import Iterable, Optional

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import glob
import math
import platform
import re
import subprocess
import time

# numpy is imported lazily within benchmark functions to keep --help usable

# --- Shared types and helpers (inlined from former scripts) ---
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


_VERSION_PATTERN = re.compile(r"-v\d+$", re.IGNORECASE)
_OBSERVATION_SUFFIXES = ("-ram", "-rgb")
_MODE_SUFFIXES = ("NoFrameskip", "Deterministic")


def detect_alepy_binary() -> AleBinaryInfo:
    import ale_py  # local import to surface ImportError clearly

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


def _camel_to_snake(name: str) -> str:
    name = name.replace("-", "_")
    name = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
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

    from ale_py import roms  # local import to surface ImportError clearly

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


def _snake_to_camel(name: str) -> str:
    parts = [p for p in name.split("_") if p]
    return "".join(part.capitalize() for part in parts)


def infer_env_id(user_spec: str, rom: ResolvedRom) -> str:
    spec = user_spec.strip()
    if "/" in spec:
        return spec
    return f"ALE/{_snake_to_camel(rom.name)}-v5"


@dataclass
class BenchmarkResult:
    frames: int
    elapsed: float
    requested: int

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed > 0 else float("inf")


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


def benchmark_alepy_vector_env(
    rom: ResolvedRom,
    frames: int,
    num_envs: int,
    *,
    num_threads: Optional[int] = None,
    warmup_steps: int = 256,
    chunk_steps: int = 256,
) -> BenchmarkResult:
    from ale_py.vector_env import AtariVectorEnv  # local import
    import numpy as np

    assert frames > 0, "Frame count must be positive"
    assert num_envs > 0, "Number of environments must be positive"
    assert chunk_steps > 0, "Chunk size must be positive"

    if rom.is_custom:
        raise ValueError(
            "Custom ROM paths are not supported by the ALE vector interface; pass a packaged ROM id like 'pong'."
        )

    threads = num_threads if num_threads is not None else num_envs
    env = AtariVectorEnv(
        game=rom.name,
        num_envs=num_envs,
        num_threads=threads,
        grayscale=True,
        stack_num=4,
        frameskip=4,
        img_height=84,
        img_width=84,
        maxpool=True,
        noop_max=0,
        repeat_action_probability=0.25,
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


def benchmark_gym_vec_env(
    env_id: str,
    rom: ResolvedRom,
    frames: int,
    num_envs: int,
    *,
    use_subproc: bool,
    warmup_steps: int = 256,
    chunk_steps: int = 256,
) -> BenchmarkResult:
    import gymnasium as gym  # local import
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
    from gymnasium.spaces import Discrete
    import numpy as np

    if rom.is_custom:
        raise ValueError(
            "Custom ROM paths are not supported by Gymnasium ALE envs; pass a packaged ROM id like 'ALE/Pong-v5'."
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
        _ = obs

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


def _run_gym(env_id: str, rom: ResolvedRom, frames: int, num_envs: int, use_subproc: bool) -> RunOutcome:
    result = benchmark_gym_vec_env(
        env_id, rom, frames=frames, num_envs=num_envs, use_subproc=use_subproc
    )
    backend = "AsyncVectorEnv" if use_subproc else "SyncVectorEnv"
    return RunOutcome(
        backend="gym",
        result=result,
        detail=f"backend={backend}, num_envs={num_envs}",
    )


def benchmark_gym_make_vec_env(
    env_id: str,
    rom: ResolvedRom,
    frames: int,
    num_envs: int,
    *,
    use_subproc: bool,
    warmup_steps: int = 256,
    chunk_steps: int = 256,
) -> BenchmarkResult:
    import gymnasium as gym  # local import
    from gymnasium.spaces import Discrete
    import numpy as np

    if rom.is_custom:
        raise ValueError(
            "Custom ROM paths are not supported by Gymnasium ALE envs; pass a packaged ROM id like 'ALE/Pong-v5'."
        )
    assert frames > 0, "Frame count must be positive"
    assert num_envs > 0, "Number of environments must be positive"
    assert chunk_steps > 0, "Chunk size must be positive"

    vec_mode = "async" if use_subproc else "sync"
    vector_kwargs = dict(
        # frameskip=1, stack_num=4, grayscale=True, img_height=84, img_width=84,
        #use_fire_reset=True, reward_clipping=True, episodic_life=False,
        #noop_max=30, repeat_action_probability=0.0
        obs_type="rgb",
    )


    if not hasattr(gym, "make_vec"):
        raise ImportError("Your Gymnasium version does not expose gym.make_vec")

    vec_env = gym.make_vec(env_id, num_envs=num_envs, vectorization_mode=vec_mode, **vector_kwargs)

    try:
        out = vec_env.reset()
        # reset() can return (obs, infos) in Gymnasium; accept both signature forms.
        if isinstance(out, tuple) and len(out) == 2:
            obs, _infos = out
        else:
            obs = out
        _ = obs

        if hasattr(vec_env, "single_action_space") and isinstance(vec_env.single_action_space, Discrete):
            action_value = 0
        elif hasattr(vec_env, "action_space") and isinstance(vec_env.action_space, Discrete):
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


def _run_gym_make(env_id: str, rom: ResolvedRom, frames: int, num_envs: int, use_subproc: bool) -> RunOutcome:
    result = benchmark_gym_make_vec_env(
        env_id, rom, frames=frames, num_envs=num_envs, use_subproc=use_subproc
    )
    backend = "async" if use_subproc else "sync"
    return RunOutcome(
        backend="gym-make",
        result=result,
        detail=f"backend=make_vec({backend}), num_envs={num_envs}",
    )


def benchmark_sb3_vec_env(
    env_id: str,
    rom: ResolvedRom,
    frames: int,
    num_envs: int,
    *,
    use_subproc: bool,
    warmup_steps: int = 256,
    chunk_steps: int = 256,
) -> BenchmarkResult:
    from stable_baselines3.common.env_util import make_vec_env  # local import
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from gymnasium.spaces import Discrete
    import numpy as np

    if rom.is_custom:
        raise ValueError(
            "Custom ROM paths are not supported by Gymnasium ALE envs; pass a packaged ROM id like 'ALE/Pong-v5'."
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
    vec_env = make_vec_env(
        env_id,
        n_envs=num_envs,
        seed=0,
        wrapper_class=None,
        monitor_dir=None,
        vec_env_cls=(SubprocVecEnv if use_subproc else DummyVecEnv),
        env_kwargs=env_kwargs,
    )

    try:
        obs = vec_env.reset()
        _ = obs
        action_space = vec_env.action_space
        if isinstance(action_space, Discrete):
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


def _run_sb3(env_id: str, rom: ResolvedRom, frames: int, num_envs: int, use_subproc: bool) -> RunOutcome:
    result = benchmark_sb3_vec_env(
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
        choices=("alepy", "gym", "gym-make", "sb3"),
        default=None,
        help=(
            "Backend to run. If omitted, runs all (alepy + gym + gym-make + sb3) sequentially "
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
    try:
        binary_info = detect_alepy_binary()
    except ImportError as e:
        binary_info = None
        print(f"ALE binary info unavailable (ale-py not installed): {e}")
    else:
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
        # Catch ImportError for soft dependencies and continue to provide useful comparison.
        # Other errors should surface to fail fast.
        try:
            outcomes.append(_run_alepy(rom, args.frames, args.num_envs, args.num_threads))
        except ImportError as e:
            print(f"Skipping alepy backend (missing dependency): {e}")

        try:
            outcomes.append(_run_gym(env_id, rom, args.frames, args.num_envs, args.subproc))
        except ImportError as e:
            print(f"Skipping gym backend (missing dependency): {e}")

        try:
            outcomes.append(_run_gym_make(env_id, rom, args.frames, args.num_envs, args.subproc))
        except ImportError as e:
            print(f"Skipping gym-make backend (missing dependency): {e}")

        try:
            outcomes.append(_run_sb3(env_id, rom, args.frames, args.num_envs, args.subproc))
        except ImportError as e:
            print(f"Skipping sb3 backend (missing dependency): {e}")

        if not outcomes:
            raise ImportError(
                "No backends available: install ale-py, gymnasium, and stable-baselines3 as needed."
            )

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
    elif args.backend == "gym-make":
        outcome = _run_gym_make(env_id, rom, args.frames, args.num_envs, args.subproc)
    elif args.backend == "sb3":
        outcome = _run_sb3(env_id, rom, args.frames, args.num_envs, args.subproc)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unknown backend: {args.backend}")

    print()
    print(_format_summary(outcome, args.num_envs))


if __name__ == "__main__":
    main()
