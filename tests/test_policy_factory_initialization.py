import math
import pytest
import torch

from utils.policy_factory import create_policy, create_actor_critic_policy


@pytest.mark.unit
@pytest.mark.parametrize("policy_type,input_shape,output_shape,obs_builder", [
    ("mlp", (4,), (5,), lambda N, shp: torch.zeros(N, shp[0])),
    ("cnn", (1, 84, 84), (6,), lambda N, shp: torch.zeros(N, int(shp[0] * shp[1] * shp[2]))),
])
@pytest.mark.parametrize("activation", [
    "relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "selu", "identity"
])
@pytest.mark.parametrize("hidden_dims", [
    (16,), (32, 16),
])
def test_create_policy_uniform_init(policy_type, input_shape, output_shape, obs_builder, activation, hidden_dims):
    """
    Policies created via the factory should start unbiased: given zero observations,
    their action distribution should be uniform (equal probabilities across actions).
    This guards against biased exploration at initialization time.
    """
    model = create_policy(
        policy_type,
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        output_shape=output_shape,
        activation=activation,
    )

    batch_size = 3
    obs = obs_builder(batch_size, input_shape)
    dist, value = model(obs)

    # Policy-only models return no value head
    assert value is None

    probs = dist.probs  # (N, A)
    assert probs.shape == (batch_size, output_shape[0])

    # Each row should be uniform: equal probs across actions and sum to 1
    row_means = probs.mean(dim=-1, keepdim=True)
    assert torch.allclose(probs, row_means.expand_as(probs), atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    # Entropy equals log(A) for a uniform categorical
    A = output_shape[0]
    expected_entropy = math.log(A)
    entropy = dist.entropy()
    assert torch.allclose(entropy, torch.full_like(entropy, expected_entropy), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("policy_type,input_shape,output_shape,obs_builder", [
    ("mlp", (4,), (5,), lambda N, shp: torch.zeros(N, shp[0])),
    ("cnn", (1, 84, 84), (6,), lambda N, shp: torch.zeros(N, int(shp[0] * shp[1] * shp[2]))),
])
@pytest.mark.parametrize("activation", [
    "relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "selu", "identity"
])
@pytest.mark.parametrize("hidden_dims", [
    (16,), (32, 16),
])
def test_create_actor_critic_uniform_init(policy_type, input_shape, output_shape, obs_builder, activation, hidden_dims):
    """
    Actor-critic policies from the factory should also be unbiased at start.
    With zero observations, the policy's distribution should be uniform and
    the value head should output zeros.
    """
    model = create_actor_critic_policy(
        policy_type,
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        output_shape=output_shape,
        activation=activation,
    )

    batch_size = 4
    obs = obs_builder(batch_size, input_shape)
    dist, values = model(obs)

    probs = dist.probs  # (N, A)
    assert probs.shape == (batch_size, output_shape[0])
    assert values.shape == (batch_size,)

    # Policy should be uniform across actions
    row_means = probs.mean(dim=-1, keepdim=True)
    assert torch.allclose(probs, row_means.expand_as(probs), atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6)

    # Value head should start at zero for zero inputs (zero bias)
    assert torch.allclose(values, torch.zeros_like(values), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("activation", [
    "relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "selu", "identity"
])
def test_cnn_variants_allow_no_hidden_actor_critic(activation):
    """CNN actor-critic should also be unbiased with no MLP hidden layers after conv trunk."""
    model = create_actor_critic_policy(
        "cnn",
        input_shape=(1, 84, 84),
        hidden_dims=(),  # no post-CNN MLP
        output_shape=(7,),
        activation=activation,
    )
    N = 2
    obs = torch.zeros(N, 1 * 84 * 84)
    dist, values = model(obs)
    probs = dist.probs
    row_means = probs.mean(dim=-1, keepdim=True)
    assert torch.allclose(probs, row_means.expand_as(probs), atol=1e-6)
    assert torch.allclose(values, torch.zeros_like(values), atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("activation", [
    "relu", "tanh", "leaky_relu", "elu", "gelu", "silu", "selu", "identity"
])
def test_cnn_variants_allow_no_hidden_policy_only(activation):
    """CNN policy-only should be unbiased with no MLP hidden layers after conv trunk."""
    model = create_policy(
        "cnn",
        input_shape=(1, 84, 84),
        hidden_dims=(),  # no post-CNN MLP
        output_shape=(5,),
        activation=activation,
    )
    N = 2
    obs = torch.zeros(N, 1 * 84 * 84)
    dist, value = model(obs)
    assert value is None
    probs = dist.probs
    row_means = probs.mean(dim=-1, keepdim=True)
    assert torch.allclose(probs, row_means.expand_as(probs), atol=1e-6)
