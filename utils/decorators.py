"""Small utility decorators for this project."""

import functools
from typing import Callable


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
