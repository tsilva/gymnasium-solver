"""Neural network utilities (MLP policy and actor-critic)."""

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .torch import compute_param_group_grad_norm, init_model_weights


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
        """Compute gradient norms for named parameter groups (default: 'all')."""
        all_params = list(self.parameters())
        return {"opt/grads/norm/all": compute_param_group_grad_norm(all_params)}


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

        # Reusable initialization
        init_model_weights(self, default_activation=activation, policy_heads=[self.policy_head])

        # Register activation hooks on backbone layers
        hooks_to_register = {}
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                hooks_to_register[f'backbone.{i}'] = layer
        self.register_activation_hooks(hooks_to_register)

    def forward(self, obs: torch.Tensor):
        x = self.backbone(obs)
        logits = self.policy_head(x)
        dist = Categorical(logits=logits)
        return dist, None  # Return None for value to maintain compatibility

    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for MLP policy components."""
        
        grad_norms = super().compute_grad_norms()

        backbone_params = list(self.backbone.parameters())
        policy_head_params = list(self.policy_head.parameters())
        
        return {
            **grad_norms,
            "opt/grads/norm/backbone": compute_param_group_grad_norm(backbone_params),
            "opt/grads/norm/policy_head": compute_param_group_grad_norm(policy_head_params),
        }

class MLPActorCritic(BaseModel):
    def __init__(
        self,
        *,
        input_shape: Union[tuple[int, ...], int],
        hidden_dims: tuple[int, ...],
        output_shape: tuple[int, ...],
        activation: str
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

        # TODO: review this
        # Reusable initialization
        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

        # Register activation hooks on backbone layers
        hooks_to_register = {}
        for i, layer in enumerate(self.backbone):
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                hooks_to_register[f'backbone.{i}'] = layer
        self.register_activation_hooks(hooks_to_register)

    def forward(self, obs: torch.Tensor):
        # Forward observation through backbone
        x = self.backbone(obs)

        # TODO: this is a hack to deal with the extra dimension provided by embeddings, need to softcode this
        if x.ndim > 2 and x.shape[1] == 1:
            x = x.squeeze(1)
        
        # Forward through policy head and get policy logits
        logits = self.policy_head(x)
  
        # Create categorical distribution from logits
        policy_dist = Categorical(logits=logits)

        # Forward through value head and get value prediction
        value_pred = self.value_head(x).squeeze(-1)

        # Return policy distribution and value prediction
        return policy_dist, value_pred

    # TODO: generalize to get_metrics
    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for MLP actor-critic components."""

        grad_norms = super().compute_grad_norms()

        backbone_params = list(self.backbone.parameters())
        policy_head_params = list(self.policy_head.parameters())
        value_head_params = list(self.value_head.parameters())

        return {
            **grad_norms,
            "opt/grads/norm/backbone": compute_param_group_grad_norm(backbone_params),
            "opt/grads/norm/policy_head": compute_param_group_grad_norm(policy_head_params),
            "opt/grads/norm/value_head": compute_param_group_grad_norm(value_head_params),
        }


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

        # Initialize weights
        init_model_weights(
            self,
            default_activation=activation,
            policy_heads=[self.policy_head],
            value_heads=[self.value_head],
        )

        # Register activation hooks on CNN and MLP layers
        hooks_to_register = {}
        for i, layer in enumerate(self.cnn):
            if isinstance(layer, nn.Conv2d):
                hooks_to_register[f'cnn.{i}'] = layer
        if hasattr(self.mlp, '__iter__'):
            for i, layer in enumerate(self.mlp):
                if isinstance(layer, nn.Linear):
                    hooks_to_register[f'mlp.{i}'] = layer
        self.register_activation_hooks(hooks_to_register)

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
        policy_dist = Categorical(logits=logits)

        # Value head
        value_pred = self.value_head(x).squeeze(-1)

        return policy_dist, value_pred

    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for CNN actor-critic components."""
        grad_norms = super().compute_grad_norms()

        cnn_params = list(self.cnn.parameters())
        mlp_params = list(self.mlp.parameters()) if hasattr(self.mlp, 'parameters') else []
        policy_head_params = list(self.policy_head.parameters())
        value_head_params = list(self.value_head.parameters())

        return {
            **grad_norms,
            "opt/grads/norm/cnn": compute_param_group_grad_norm(cnn_params),
            "opt/grads/norm/mlp": compute_param_group_grad_norm(mlp_params),
            "opt/grads/norm/policy_head": compute_param_group_grad_norm(policy_head_params),
            "opt/grads/norm/value_head": compute_param_group_grad_norm(value_head_params),
        }
