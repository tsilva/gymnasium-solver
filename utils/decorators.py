"""
Utility decorators for the gymnasium-solver project.
"""

import functools
from typing import Callable, Any


def must_implement(func: Callable) -> Callable:
    """
    Decorator to mark methods that must be implemented by subclasses.
    
    This decorator replaces the pattern of having methods that just raise NotImplementedError.
    Instead, the method can be defined normally and decorated with @must_implement.
    
    Usage:
        class BaseClass:
            @must_implement
            def some_method(self):
                pass  # or provide default implementation
    
    Args:
        func: The method to be decorated
        
    Returns:
        The decorated method that raises NotImplementedError when called
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        class_name = args[0].__class__.__name__ if args else "Unknown"
        method_name = func.__name__
        raise NotImplementedError(
            f"Subclass {class_name} must implement {method_name}()"
        )
    
    return wrapper
