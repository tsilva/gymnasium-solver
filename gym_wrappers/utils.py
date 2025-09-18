"""Shared utilities for working with Gymnasium wrapper chains."""
from __future__ import annotations

from typing import Iterator, Optional, Type, TypeVar

import gymnasium as gym

EnvType = TypeVar("EnvType", bound=gym.Env)


def _iter_env_chain(root: gym.Env) -> Iterator[gym.Env]:
    """Yield wrappers from ``root`` down to the innermost environment."""
    current: Optional[gym.Env] = root
    while isinstance(current, gym.Env):
        yield current
        if not hasattr(current, "env"):
            break
        next_env = getattr(current, "env")  # type: ignore[attr-defined]
        if next_env is None or next_env is current:
            break
        current = next_env


def find_wrapper(root: gym.Env, wrapper_type: Type[EnvType]) -> Optional[EnvType]:
    """Return the first wrapper of ``wrapper_type`` encountered in the chain."""
    for wrapper in _iter_env_chain(root):
        if isinstance(wrapper, wrapper_type):
            return wrapper
    return None


def get_innermost_env(root: gym.Env) -> gym.Env:
    """Return the innermost environment in the wrapper chain starting at ``root``."""
    innermost = root
    for wrapper in _iter_env_chain(root):
        innermost = wrapper
    return innermost
