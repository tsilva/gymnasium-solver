"""Model architecture registry for predefined model configurations."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class ModelSpec:
    """Specification for a model architecture preset."""
    policy: str  # "mlp", "mlp_actorcritic", "cnn_actorcritic", etc
    hidden_dims: Tuple[int, ...]
    activation: str = "relu"
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)


# Registry of predefined model architectures
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # ============================================================================
    # MLP Models (for vector observations)
    # ============================================================================
    "mlp_tiny": ModelSpec(
        policy="mlp_actorcritic",
        hidden_dims=(64,),
    ),
    "mlp_small": ModelSpec(
        policy="mlp_actorcritic",
        hidden_dims=(128, 128),
    ),
    "mlp_medium": ModelSpec(
        policy="mlp_actorcritic",
        hidden_dims=(256, 256),
    ),
    "mlp_large": ModelSpec(
        policy="mlp_actorcritic",
        hidden_dims=(512, 512),
    ),

    # ============================================================================
    # CNN Models (for image observations)
    # ============================================================================

    # Nature DQN architecture (Mnih et al. 2015)
    # Standard for Atari: 32→64→64 channels, 8×8→4×4→3×3 kernels
    "cnn_nature": ModelSpec(
        policy="cnn_actorcritic",
        hidden_dims=(512,),
        policy_kwargs={
            "channels": (32, 64, 64),
            "kernel_sizes": (8, 4, 3),
            "strides": (4, 2, 1),
        }
    ),

    # IMPALA-style smaller CNN (Espeholt et al. 2018)
    # Lighter weight: 16→32→32 channels
    "cnn_impala": ModelSpec(
        policy="cnn_actorcritic",
        hidden_dims=(256,),
        policy_kwargs={
            "channels": (16, 32, 32),
            "kernel_sizes": (8, 4, 3),
            "strides": (4, 2, 1),
        }
    ),

    # Larger capacity CNN for complex visual tasks
    "cnn_large": ModelSpec(
        policy="cnn_actorcritic",
        hidden_dims=(1024,),
        policy_kwargs={
            "channels": (32, 64, 128),
            "kernel_sizes": (8, 4, 3),
            "strides": (4, 2, 1),
        }
    ),
}


def resolve_model_spec(model_id: str) -> ModelSpec:
    """Resolve model_id to a ModelSpec.

    Args:
        model_id: Name of the model preset (e.g., "mlp_small", "cnn_nature")

    Returns:
        ModelSpec instance with architecture details

    Raises:
        AssertionError: If model_id is not in registry
    """
    assert model_id in MODEL_REGISTRY, \
        f"Unknown model_id: '{model_id}'. Available models: {sorted(MODEL_REGISTRY.keys())}"
    return MODEL_REGISTRY[model_id]


def list_models() -> list[str]:
    """Return list of available model IDs."""
    return sorted(MODEL_REGISTRY.keys())
