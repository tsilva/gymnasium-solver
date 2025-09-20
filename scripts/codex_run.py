#!/usr/bin/env python3
"""Run Codex CLI with a one-off prompt (Python wrapper)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ARGS = ["-m", "gpt-5", "-c", "model_reasoning_effort=high"]


def read_stdin() -> str:
    if sys.stdin is None or sys.stdin.closed:
        return ""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Codex CLI with a one-off prompt")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--prompt", "-p", type=str, help="Prompt text")
    g.add_argument("--file", "-f", type=Path, help="Read prompt from file")
    args = parser.parse_args()

    if shutil.which("codex") is None:
        print("Error: `codex` CLI not found on PATH.", file=sys.stderr)
        return 127

    prompt = ""
    if args.prompt:
        prompt = args.prompt
    elif args.file:
        try:
            prompt = args.file.read_text()
        except (OSError, UnicodeError) as e:
            print(f"Failed to read {args.file}: {e}", file=sys.stderr)
            return 2
    else:
        prompt = read_stdin()

    if not prompt.strip():
        print("Empty prompt. Provide --prompt, --file, or pipe via STDIN.", file=sys.stderr)
        return 3

    try:
        proc = subprocess.run(
            ["codex", *DEFAULT_ARGS],
            input=prompt,
            text=True,
            check=False,
        )
        return proc.returncode
    except FileNotFoundError:
        print("Error: `codex` CLI not found on PATH.", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
