"""Small utility decorators for this project."""

import functools
from typing import Any, Callable, Dict, Tuple


def must_implement(func: Callable) -> Callable:
    """Decorator marking methods that subclasses must implement."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else "Unknown"
        method_name = func.__name__
        raise NotImplementedError(
            f"Subclass {class_name} must implement {method_name}()"
        )

    return wrapper


def cache(func: Callable) -> Callable:
    """Cache decorator for instance methods with stable arguments."""

    memo: Dict[Tuple[str, Tuple[Any, ...], frozenset], Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (func.__name__, args, frozenset(kwargs.items()))
        if key in memo:
            return memo[key]

        result = func(*args, **kwargs)
        memo[key] = result
        return result

    return wrapper
