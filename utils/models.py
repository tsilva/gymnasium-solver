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


class BaseModel(nn.Module):
    """Base module with gradient-norm reporting (single 'all' group by default)."""

    def compute_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for named parameter groups (default: 'all')."""
        all_params = list(self.parameters())
        return {"grad_norm/all": compute_param_group_grad_norm(all_params)}


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
            "grad_norm/backbone": compute_param_group_grad_norm(backbone_params),
            "grad_norm/policy_head": compute_param_group_grad_norm(policy_head_params),
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
            "grad_norm/backbone": compute_param_group_grad_norm(backbone_params),
            "grad_norm/policy_head": compute_param_group_grad_norm(policy_head_params),
            "grad_norm/value_head": compute_param_group_grad_norm(value_head_params),
        }
