"""Environment setup utilities."""

import warnings
from typing import Callable, Any
from tsilva_notebook_utils.gymnasium import build_env as _build_env, set_random_seed


def suppress_warnings():
    """Suppress common warnings that clutter notebook output."""
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
    # Suppress pygame-specific pkg_resources warnings
    warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)
    # Suppress NSXPCSharedListener warnings on macOS
    warnings.filterwarnings("ignore", message=".*NSXPCSharedListener.*")
    # General pygame warnings suppression
    warnings.filterwarnings("ignore", category=UserWarning, module="pygame.*")


def create_env_builder(config) -> Callable:
    """Create environment builder function with config parameters."""
    def build_env_fn(seed, n_envs=None):
        return _build_env(
            config.env_id,
            norm_obs=config.normalize,
            n_envs=n_envs if n_envs is not None else config.n_envs,
            seed=seed
        )
    return build_env_fn


def setup_environment(config):
    """Setup environment with configuration."""
    # Set random seed for reproducibility
    set_random_seed(config.seed)
    
    # Create environment builder
    build_env_fn = create_env_builder(config)
    
    return build_env_fn
