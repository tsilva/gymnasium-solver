"""Neural network utilities (MLP policy and actor-critic)."""

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli, Independent

from .torch import compute_param_group_grad_norm, init_model_weights
from .distributions import MaskedCategorical
from .policy_ops import create_action_distribution


def resolve_activation(activation_id: str) -> type[nn.Module]:
    key = activation_id.lower()
    mapping = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,  # alias
        "identity": nn.Identity,
    }
    return mapping[key]


def build_mlp(input_shape: Union[tuple[int, ...], int], hidden_dims: tuple[int, ...], activation:str):
    """Build an MLP stack with the given activation."""
    is_int = type(input_shape) in [int, np.int32, np.int64]
    assert is_int or len(input_shape) == 1, "Input shape must be 1D"

    # Resolve the activation function id (string) to a class
    activation_cls = resolve_activation(activation)

    # Create a list of layers
    layers = []

    # If input shape is an int, use an
    # embedding layer to convert int to vector
    _hidden_dims = hidden_dims
    if is_int:
        n_embeddings = input_shape
        embedding_dim = hidden_dims[0]
        layers += [nn.Embedding(n_embeddings, embedding_dim)]
        last_dim = embedding_dim
        _hidden_dims = hidden_dims[1:]
    # Otherwise, use the first dimension of the input shape
    else:
        last_dim = input_shape[0]

    # Create the MLP layers
    for _hidden_dim in _hidden_dims:
        layers += [nn.Linear(last_dim, _hidden_dim), activation_cls()]
        last_dim = _hidden_dim

    # Stack the layers into a sequential model
    model = nn.Sequential(*layers)

    # Return the model
    return model


def build_cnn(
    input_shape: tuple[int, ...],
    channels: tuple[int, ...] = (32, 64, 64),
    kernel_sizes: tuple[int, ...] = (8, 4, 3),
    strides: tuple[int, ...] = (4, 2, 1),
    activation: str = "relu",
    output_dim: int | None = None,
) -> tuple[nn.Module, int]:
    """Build a CNN feature extractor.

    Args:
        input_shape: Input shape (C, H, W) for image observations
        channels: Number of output channels for each conv layer
        kernel_sizes: Kernel size for each conv layer
        strides: Stride for each conv layer
        activation: Activation function name
        output_dim: If provided, add a Linear layer to project to this dim

    Returns:
        Tuple of (cnn_module, output_features) where output_features is the
        flattened feature dimension after convolutions (or output_dim if specified)
    """
    assert len(input_shape) == 3, f"CNN input must be 3D (C, H, W), got {input_shape}"
    assert len(channels) == len(kernel_sizes) == len(strides), \
        "channels, kernel_sizes, and strides must have same length"

    activation_cls = resolve_activation(activation)

    layers = []
    in_channels = input_shape[0]

    # Build convolutional layers
    for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        layers.append(activation_cls())
        in_channels = out_channels

    # Add flatten layer
    layers.append(nn.Flatten())

    # Compute output size by doing a forward pass
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        cnn_partial = nn.Sequential(*layers)
        features = cnn_partial(dummy_input)
        n_features = features.shape[1]

    # Optionally add linear projection
    if output_dim is not None:
        layers.append(nn.Linear(n_features, output_dim))
        layers.append(activation_cls())
        n_features = output_dim

    cnn = nn.Sequential(*layers)
    return cnn, n_features


class BaseModel(nn.Module):
    """Base module with gradient-norm reporting (single 'all' group by default)."""

    def __init__(self):
        super().__init__()
        self._activation_stats = {}
        self._track_activations = False

    def _make_activation_hook(self, name: str):
        """Create a forward hook to capture activation statistics."""
        def hook(module, input, output):
            if not self._track_activations:
                return

            with torch.no_grad():
                # Handle both tensor and tuple outputs
                if isinstance(output, tuple):
                    output = output[0]

                # Flatten to compute statistics across all dimensions except batch
                flat = output.flatten(start_dim=1)

                # Compute dead neuron percentage (activations near zero)
                dead_threshold = 1e-6
                dead_mask = flat.abs() < dead_threshold
                dead_pct = dead_mask.float().mean(dim=0)  # Per-neuron across batch

                self._activation_stats[name] = {
                    'mean': flat.mean().item(),
                    'std': flat.std().item(),
                    'dead_pct': dead_pct.mean().item(),  # Average across neurons
                    'dead_max': dead_pct.max().item(),   # Max dead percentage
                }

        return hook

    def register_activation_hooks(self, module_dict: Dict[str, nn.Module]):
        """Register activation hooks on specified modules.

        Args:
            module_dict: Dict mapping names to modules to track
        """
        for name, module in module_dict.items():
            module.register_forward_hook(self._make_activation_hook(name))

    def register_backbone_hooks(self, backbone: nn.Module, prefix: str = 'backbone', layer_types: tuple = None):
        """Register activation hooks on backbone layers.

        Args:
            backbone: The backbone module (Sequential or ModuleList)
            prefix: Prefix for hook names (default: 'backbone')
            layer_types: Tuple of layer types to track (default: Linear, Embedding, Conv2d)
        """
        if layer_types is None:
            layer_types = (nn.Linear, nn.Embedding, nn.Conv2d)

        hooks_to_register = {}
        if isinstance(backbone, (nn.Sequential, nn.ModuleList)) or hasattr(backbone, '__iter__'):
            for i, layer in enumerate(backbone):
                if isinstance(layer, layer_types):
                    hooks_to_register[f'{prefix}.{i}'] = layer
        self.register_activation_hooks(hooks_to_register)

    def compute_activation_stats(self) -> Dict[str, float]:
        """Compute activation statistics from stored activations.

        Returns dict of metric_name -> value for logging.
        """
        if not self._activation_stats:
            return {}

        metrics = {}
        for name, stats in self._activation_stats.items():
            metrics[f'opt/activations/{name}/mean'] = stats['mean']
            metrics[f'opt/activations/{name}/std'] = stats['std']
            metrics[f'opt/activations/{name}/dead_pct'] = stats['dead_pct']
            metrics[f'opt/activations/{name}/dead_max'] = stats['dead_max']

        # Clear stats after reading
        self._activation_stats.clear()

        return metrics

    def compute_grad_norms(self) -> Dict[str, float]:
        """Auto-discover and compute grad norms for standard components."""
        components = {}

        # Auto-discover by naming convention
        for name in ['backbone', 'trunk', 'cnn', 'mlp', 'policy_head', 'actor_head', 'value_head', 'critic_head']:
            if hasattr(self, name):
                module = getattr(self, name)
                if isinstance(module, nn.Module):
                    components[name] = list(module.parameters())

        # If no components found, fall back to all parameters
        if not components:
            all_params = list(self.parameters())
            return {"opt/grads/norm/all": compute_param_group_grad_norm(all_params)}

        return self.compute_component_grad_norms(components)

    def compute_component_grad_norms(self, components: Dict[str, list]) -> Dict[str, float]:
        """Compute gradient norms for base + named components.

        Args:
            components: Dict mapping component names to parameter lists

        Returns:
            Dict of "opt/grads/norm/{name}" -> norm value
        """
        grad_norms = {
            "opt/grads/norm/all": compute_param_group_grad_norm(list(self.parameters()))
        }

        for name, params in components.items():
            grad_norms[f"opt/grads/norm/{name}"] = compute_param_group_grad_norm(params)

        return grad_norms


class MLPPolicy(BaseModel):
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (64,),
        output_dim: int | None = None,
        activation: str = "relu",
        *,
        # Back-compat and factory-compat kwargs
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        valid_actions: list[int] | None = None,
        action_space_type: str = "discrete",  # "discrete" or "multibinary"
    ):
        super().__init__()

        # Harmonize input/output dims to support both direct ints and shape tuples
        if input_dim is None:
            assert input_shape is not None and len(input_shape) == 1, "Input shape must be 1D"
            input_dim = int(input_shape[0])
        if output_dim is None:
            assert output_shape is not None and len(output_shape) == 1, "Output shape must be 1D"
            output_dim = int(output_shape[0])

        # Normalize hidden dims and create MLP backbone
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        self.backbone = build_mlp((input_dim,), hidden_dims, activation)

        # Create the policy head
        self.policy_head = nn.Linear(hidden_dims[-1], output_dim)

        # Store valid actions for action masking (if specified)
        self.valid_actions = valid_actions
        self.action_space_type = action_space_type

        # Reusable initialization
        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

        # Register activation hooks on backbone layers
        self.register_backbone_hooks(self.backbone)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)

        # Create distribution with optional action masking
        dist = create_action_distribution(logits, self.valid_actions, self.action_space_type)

        return dist, None  # Return None for value to maintain compatibility


class MLPActorCritic(BaseModel):
    def __init__(
        self,
        *,
        input_shape: Union[tuple[int, ...], int],
        hidden_dims: tuple[int, ...],
        output_shape: tuple[int, ...],
        activation: str,
        valid_actions: list[int] | None = None,
        action_space_type: str = "discrete",  # "discrete" or "multibinary"
    ):
        super().__init__()
        assert type(input_shape) in [int, np.int32, np.int64] or len(input_shape) == 1, "Input shape must be 1D"
        assert len(output_shape) == 1, "Output shape must be 1D"

        self.backbone = build_mlp(input_shape, hidden_dims, activation)

        # Create the policy head
        policy_input_dim = hidden_dims[-1]
        policy_output_dim = output_shape[0]
        self.policy_head = nn.Linear(policy_input_dim, policy_output_dim)

        # Create the value head
        value_input_dim = hidden_dims[-1]
        value_output_dim = 1
        self.value_head = nn.Linear(value_input_dim, value_output_dim)

        # Store valid actions for action masking (if specified)
        self.valid_actions = valid_actions
        self.action_space_type = action_space_type

        # TODO: review this
        # Reusable initialization
        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

        # Register activation hooks on backbone layers
        self.register_backbone_hooks(self.backbone)

    def forward(self, obs: torch.Tensor):
        # Forward observation through backbone
        x = self.backbone(obs)

        # TODO: this is a hack to deal with the extra dimension provided by embeddings, need to softcode this
        if x.ndim > 2 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Forward through policy head and get policy logits
        logits = self.policy_head(x)

        # Create distribution with optional action masking
        policy_dist = create_action_distribution(logits, self.valid_actions, self.action_space_type)

        # Forward through value head and get value prediction
        value_pred = self.value_head(x).squeeze(-1)

        # Return policy distribution and value prediction
        return policy_dist, value_pred



class CNNActorCritic(BaseModel):
    """CNN-based actor-critic for image observations."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, ...],
        hidden_dims: tuple[int, ...] = (512,),
        output_shape: tuple[int, ...],
        activation: str = "relu",
        channels: tuple[int, ...] = (32, 64, 64),
        kernel_sizes: tuple[int, ...] = (8, 4, 3),
        strides: tuple[int, ...] = (4, 2, 1),
        valid_actions: list[int] | None = None,
        action_space_type: str = "discrete",  # "discrete" or "multibinary"
    ):
        """Initialize CNN actor-critic.

        Args:
            input_shape: Input shape (C, H, W) for image observations
            hidden_dims: Hidden dimensions for MLP after CNN features
            output_shape: Output shape (action_dim,)
            activation: Activation function name
            channels: CNN channel sizes
            kernel_sizes: CNN kernel sizes
            strides: CNN strides
            valid_actions: Optional list of valid action indices for masking
        """
        super().__init__()
        assert len(input_shape) == 3, f"CNN input must be 3D (C, H, W), got {input_shape}"
        assert len(output_shape) == 1, "Output shape must be 1D"

        # Build CNN feature extractor
        self.cnn, cnn_features = build_cnn(
            input_shape=input_shape,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation=activation,
            output_dim=None,  # Don't project yet, use MLP
        )

        # Build MLP on top of CNN features
        if hidden_dims:
            self.mlp = build_mlp((cnn_features,), hidden_dims, activation)
            mlp_output_dim = hidden_dims[-1]
        else:
            # No MLP, use CNN features directly
            self.mlp = nn.Identity()
            mlp_output_dim = cnn_features

        # Policy head
        self.policy_head = nn.Linear(mlp_output_dim, output_shape[0])

        # Value head
        self.value_head = nn.Linear(mlp_output_dim, 1)

        # Store valid actions for action masking (if specified)
        self.valid_actions = valid_actions
        self.action_space_type = action_space_type

        # Initialize weights
        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

        # Register activation hooks on CNN and MLP layers
        self.register_backbone_hooks(self.cnn, prefix='cnn')
        if hasattr(self.mlp, '__iter__'):
            self.register_backbone_hooks(self.mlp, prefix='mlp')

    def forward(self, obs: torch.Tensor):
        """Forward pass through CNN, MLP, and heads.

        Args:
            obs: Observations of shape (batch, C, H, W)

        Returns:
            Tuple of (policy_dist, value_pred)
        """
        # Normalize observations to [0, 1] range
        # Handle both uint8 and float observations
        if obs.dtype in (torch.uint8, torch.int8):
            obs = obs.to(torch.float32) / 255.0
        elif obs.dtype != torch.float32:
            obs = obs.to(torch.float32)

        # Extract CNN features
        features = self.cnn(obs)

        # Pass through MLP
        x = self.mlp(features)

        # Policy head
        logits = self.policy_head(x)

        # Create distribution with optional action masking
        policy_dist = create_action_distribution(logits, self.valid_actions, self.action_space_type)

        # Value head
        value_pred = self.value_head(x).squeeze(-1)

        return policy_dist, value_pred
