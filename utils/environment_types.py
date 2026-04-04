"""Environment family detection helpers."""

from __future__ import annotations


ENV_TYPE_PATTERNS = {
    "alepy": lambda env_id: env_id.lower().startswith("ale/"),
    "vizdoom": lambda env_id: env_id.lower().startswith("vizdoom-"),
    "stable_retro": lambda env_id: env_id.lower().startswith("retro/"),
    "mab": lambda env_id: (
        env_id.lower().startswith("bandit-")
        or env_id.lower().startswith("bandit/")
        or env_id.lower() == "bandit-v0"
    ),
}


def get_env_type(env_id: str) -> str | None:
    """Return env type or None if standard Gymnasium environment."""
    for env_type, matcher in ENV_TYPE_PATTERNS.items():
        if matcher(env_id):
            return env_type
    return None
