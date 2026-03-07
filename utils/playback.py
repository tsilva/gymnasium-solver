"""Shared helpers for playback and inspection entrypoints."""

from __future__ import annotations

from typing import Any


def resolve_playback_seed(config: Any, seed_spec: str | int | None) -> int:
    """Resolve playback seed from CLI-friendly values."""
    if seed_spec is None:
        return int(config.seed_test)
    if seed_spec in {"train", "val", "test"}:
        return int(
            {
                "train": config.seed_train,
                "val": config.seed_val,
                "test": config.seed_test,
            }[seed_spec]
        )
    return int(seed_spec)


def get_action_labels_from_env(env: Any) -> dict[int, str] | None:
    """Return action labels from the environment as an integer-keyed dict."""
    if not hasattr(env, "get_action_labels"):
        return None

    raw_labels = env.get_action_labels()
    if raw_labels is None:
        return None

    if isinstance(raw_labels, dict):
        labels: dict[int, str] = {}
        for key, value in raw_labels.items():
            try:
                labels[int(key)] = str(value)
            except (TypeError, ValueError):
                return None
        return labels or None

    labels = {index: str(value) for index, value in enumerate(raw_labels)}
    return labels or None


def get_action_label_list_from_env(env: Any) -> list[str] | None:
    """Return action labels from the environment as a dense list."""
    labels = get_action_labels_from_env(env)
    if not labels:
        return None

    max_idx = max(labels)
    return [labels.get(index, str(index)) for index in range(max_idx + 1)]


def build_single_env_from_config(config: Any, *, render_mode: str | None, seed: int):
    """Build a single synchronous environment for playback-style flows."""
    from utils.environment import build_env_from_config

    return build_env_from_config(
        config,
        n_envs=1,
        vectorization_mode="sync",
        render_mode=render_mode,
        seed=seed,
    )
